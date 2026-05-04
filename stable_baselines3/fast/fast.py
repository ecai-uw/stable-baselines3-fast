import os
from copy import deepcopy
from typing import Any, ClassVar, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from omegaconf import OmegaConf
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, get_schedule_fn, polyak_update, should_collect_more_steps, update_learning_rate
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor, FlattenExtractor, NatureCNN
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.fast.policies import BaseCriticValue, ResidualSACPolicy

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from stable_baselines3.fast.buffers import FastBuffer, DictFastBuffer

from tqdm import tqdm
from functools import partial
import warnings

SelfFAST = TypeVar("SelfFAST", bound="FAST")


def _add_gaussian(x, std):
    """Additive iid Gaussian noise. Dispatches on np.ndarray vs torch.Tensor."""
    if isinstance(x, th.Tensor):
        return x + th.randn_like(x) * std
    return x + (np.random.standard_normal(x.shape) * std).astype(x.dtype, copy=False)


def _quat_tangent_noise(quat, sigma_rad):
    """Small-rotation noise on a unit quaternion in xyzw convention.

    Samples axis-angle delta ~ N(0, sigma_rad^2 * I_3), composes Δq ⊗ q via
    Hamilton product, renormalizes. Operates over the trailing dim-4 axis;
    leading dims (batch / n_envs) pass through. Dispatches np vs torch.
    """
    if isinstance(quat, th.Tensor):
        delta = th.randn(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype) * sigma_rad
        theta = th.linalg.norm(delta, dim=-1, keepdim=True)
        half = 0.5 * theta
        small = theta < 1e-8
        safe_theta = th.where(small, th.ones_like(theta), theta)
        ratio = th.where(small, th.full_like(theta, 0.5), th.sin(half) / safe_theta)
        dq_w = th.cos(half)
        cat = lambda xs: th.cat(xs, dim=-1)
        norm = lambda x: th.linalg.norm(x, dim=-1, keepdim=True)
    else:
        delta = (np.random.standard_normal(quat.shape[:-1] + (3,)) * sigma_rad).astype(quat.dtype, copy=False)
        theta = np.linalg.norm(delta, axis=-1, keepdims=True)
        half = 0.5 * theta
        small = theta < 1e-8
        safe_theta = np.where(small, np.ones_like(theta), theta)
        ratio = np.where(small, np.full_like(theta, 0.5), np.sin(half) / safe_theta)
        dq_w = np.cos(half)
        cat = lambda xs: np.concatenate(xs, axis=-1)
        norm = lambda x: np.linalg.norm(x, axis=-1, keepdims=True)
    dq_xyz = ratio * delta
    qx, qy, qz, qw = quat[..., 0:1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:4]
    dx, dy, dz = dq_xyz[..., 0:1], dq_xyz[..., 1:2], dq_xyz[..., 2:3]
    dw = dq_w
    # Hamilton product Δq ⊗ q (xyzw):
    ox = dw * qx + dx * qw + dy * qz - dz * qy
    oy = dw * qy - dx * qz + dy * qw + dz * qx
    oz = dw * qz + dx * qy - dy * qx + dz * qw
    ow = dw * qw - dx * qx - dy * qy - dz * qz
    out = cat([ox, oy, oz, ow])
    return out / norm(out)


class FAST(OffPolicyAlgorithm):
    """
    FAST

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
	:param learning_rate: learning rate for adam optimizer,
		the same learning rate will be used for all networks (Q-Values, Actor and Value function)
		it can be a function of the current progress remaining (from 1 to 0)
	:param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
	:param batch_size: Minibatch size for each gradient update
	:param tau: the soft update coefficient ("Polyak update", between 0 and 1)
	:param gamma: the discount factor
	:param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
		like ``(5, "step")`` or ``(2, "episode")``.
        :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
		Set to ``-1`` means to do as many gradient steps as steps done in the environment
		during the rollout.
	:param action_noise: the action noise type (None by default), this can help
		for hard exploration problem. Cf common.noise for the different action noise type.
	:param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
		If ``None``, it will be automatically selected.
	:param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
	:param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
		at a cost of more complexity.
		See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
	:param ent_coef: Entropy regularization coefficient. (Equivalent to
		inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
		Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
	:param target_update_interval: update the target network every ``target_network_update_freq``
		gradient steps.
	:param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
	:param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
		instead of action noise exploration (default: False)
	:param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
		Default: -1 (only sample at the beginning of the rollout)
	:param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
		during the warm up phase (before learning starts)
	:param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
		the reported success rate, mean episode length, and mean reward over
	:param tensorboard_log: the log location for tensorboard (if None, no logging)
	:param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
	:param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
		debug messages
	:param seed: Seed for the pseudo random generators
	:param device: Device (cpu, cuda, ...) on which the code should be run.
		Setting it to auto, the code will be run on the GPU if possible.
        :param _init_setup_model: Whether or not to build the network at the creation of the instance
	:param actor_gradient_steps: Number of gradient steps to take on actor per training update
	:param diffusion_policy: The diffusion policy to use for action generation
	:param critic_backup_combine_type: How to combine the critics for the backup (min or mean)
    """
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
		"MlpPolicy": MlpPolicy,
		"CnnPolicy": CnnPolicy,
		"MultiInputPolicy": MultiInputPolicy,
	}
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
		self,
		policy: Union[str, type[SACPolicy]],
		env: Union[GymEnv, str],
		learning_rate: Union[float, Schedule] = 3e-4,
		buffer_size: int = 1_000_000,  # 1e6
		learning_starts: int = 100,
		batch_size: int = 256,
		tau: float = 0.005,
		gamma: float = 0.99,
		train_freq: Union[int, tuple[int, str]] = 1,
		gradient_steps: int = 1,
		action_noise: Optional[ActionNoise] = None,
		replay_buffer_class: Optional[type[ReplayBuffer]] = None,
		replay_buffer_kwargs: Optional[dict[str, Any]] = None,
		optimize_memory_usage: bool = False,
		ent_coef: Union[str, float] = "auto",
		target_update_interval: int = 1,
		target_entropy: Union[str, float] = "auto",
		use_sde: bool = False,
		sde_sample_freq: int = -1,
		use_sde_at_warmup: bool = False,
		stats_window_size: int = 100,
		tensorboard_log: Optional[str] = None,
		policy_kwargs: Optional[dict[str, Any]] = None,
		verbose: int = 0,
		seed: Optional[int] = None,
		device: Union[th.device, str] = "auto",
		_init_setup_model: bool = True,
		actor_gradient_steps: int = -1,
		diffusion_policy=None,
        critic_backup_combine_type='min',
        critic_lr: Optional[Union[float, Schedule]] = None,
        cfg: dict = {},
	):
        super().__init__(
			policy,
			env,
			learning_rate,
			buffer_size,
			learning_starts,
			batch_size,
			tau,
			gamma,
			train_freq,
			gradient_steps,
			action_noise,
			replay_buffer_class=replay_buffer_class,
			replay_buffer_kwargs=replay_buffer_kwargs,
			policy_kwargs=policy_kwargs,
			stats_window_size=stats_window_size,
			tensorboard_log=tensorboard_log,
			verbose=verbose,
			device=device,
			seed=seed,
			use_sde=use_sde,
			sde_sample_freq=sde_sample_freq,
			use_sde_at_warmup=use_sde_at_warmup,
			optimize_memory_usage=optimize_memory_usage,
			supported_action_spaces=(spaces.Box,),
			support_multi_env=True,
		)

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
		# Entropy coefficient / Entropy temperature
		# Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.actor_gradient_steps = actor_gradient_steps
        # Separate critic LR (also applied to value-net optimizer). None = use learning_rate.
        self._critic_lr_input = critic_lr

        # TODO: clean up; abstract as many params as possible into this cfg
        self.cfg = cfg

        self.diffusion_policy = diffusion_policy
        self.critic_backup_combine_type = critic_backup_combine_type
        # Diagnostic toggle: drop the SAC entropy bonus from the critic backup
        # while leaving actor / α-autotune unchanged. Defaults to True for
        # backwards compatibility with older configs.
        self.critic_entropy_bonus = bool(self.cfg.train.get("critic_entropy_bonus", True)) if self.cfg else True
        # REDQ-style critic ensemble: target uses min over a random subset of size
        # min_q_heads. Actor loss aggregation is controlled by policy_gradient_type;
        # None falls back to the legacy critic_backup_combine_type behavior.
        # Currently the only supported non-None value is "ensemble_mean".
        self.min_q_heads = self.cfg.train.get("min_q_heads", None) if self.cfg else None
        self.policy_gradient_type = self.cfg.train.get("policy_gradient_type", None) if self.cfg else None
        if self.policy_gradient_type is not None and self.policy_gradient_type != "ensemble_mean":
            raise ValueError(
                f"Unsupported policy_gradient_type={self.policy_gradient_type!r}; "
                f"valid options are None or 'ensemble_mean'."
            )
        # Optional gradient clipping (max-norm) applied before each optimizer.step().
        self.grad_clip_norm = self.cfg.train.get("grad_clip_norm", None) if self.cfg else None

        # This will only be used for jumpstart curriculum, and will be overwritten by load mdel.
        self.jumpstart_stage = 1

        # Per-env ephemeral state for jumpstart="random"; lazy-seeded in collect_rollouts
        # and excluded from save so resumed old checkpoints simply re-seed on next rollout.
        self._random_thresholds: Optional[np.ndarray] = None
        self._prev_dones: Optional[np.ndarray] = None

        if _init_setup_model:
            self._setup_model()
        
    def _setup_model(self) -> None:
        # Extracting base policy config params.
        self.chunk_size = self.cfg.base_policy.chunk_size
        self.action_dim = self.cfg.base_policy.action_dim

        # Whether RL emits a single-step residual (with the base policy still chunked).
        # When False, RL emits a chunk_size-length residual, matching legacy behavior.
        self.rl_single_step = bool(self.cfg.rl_single_step) if "rl_single_step" in self.cfg else False

        # How many upcoming base-policy actions the residual sees per decision.
        # Lives under cfg.policy because it shapes the residual's input — the base
        # policy itself is indifferent. null/missing => backcompat default: chunked
        # => chunk_size, single-step => 1 (the latter preserves the legacy
        # single-step behavior where the residual only saw one base action). Set
        # explicitly (typically == chunk_size or larger, up to prediction_horizon)
        # to give the single-step residual the same lookahead context the chunked
        # path already enjoys.
        cfg_lookahead_k = self.cfg.policy.get("lookahead_k", None)
        if cfg_lookahead_k is None:
            self.lookahead_k = 1 if self.rl_single_step else self.chunk_size
        else:
            self.lookahead_k = int(cfg_lookahead_k)

        # When True, the critic's first layer is widened by base_action_dim so
        # Q conditions on the lookahead window in addition to the residual action.
        # Gated (default False) because flipping it on changes the critic's input
        # shape and breaks loading of any critic weights saved without it.
        self.critic_use_base_action = bool(self.cfg.policy.get("critic_use_base_action", False))

        # Full base-policy prediction horizon. Read from cfg (not the wrapper) so it's
        # available during _setup_model — diffusion_policy is attached later in
        # train_fast.py, and FAST.load runs _setup_model before that attachment too.
        # Sized so the single-step lookahead cache can hold prediction_horizon slots
        # and vend any k-action window starting in [0, chunk_size-1] without an extra
        # base call. Defaults to chunk_size (legacy: no slack) when absent.
        self.prediction_horizon = int(self.cfg.base_policy.get("prediction_horizon", self.chunk_size))

        # Validate: only single-step uses peek_lookahead_actions / the deep cache.
        # The chunked path always feeds the residual the full chunk_size chunk and
        # ignores lookahead_k. In single-step a peek at step_idx ∈ [0, chunk_size-1]
        # must read lookahead_k slots without exceeding the cache, so the cache
        # must be at least chunk_size + lookahead_k - 1 deep.
        if self.rl_single_step and self.lookahead_k > 1:
            required_horizon = self.chunk_size + self.lookahead_k - 1
            if self.prediction_horizon < required_horizon:
                raise ValueError(
                    f"lookahead_k={self.lookahead_k} with chunk_size={self.chunk_size} requires "
                    f"prediction_horizon >= {required_horizon}, but base policy reports "
                    f"prediction_horizon={self.prediction_horizon}. Either lower lookahead_k or "
                    f"retrain the base policy with a longer chunk."
                )
        # In chunked mode the residual already sees the full chunk; lookahead_k is a
        # no-op and any other value silently mis-sizes the buffer / actor input.
        # Reject early so the misunderstanding surfaces at config time.
        if not self.rl_single_step and self.lookahead_k != self.chunk_size:
            raise ValueError(
                f"lookahead_k={self.lookahead_k} is only meaningful under rl_single_step=True. "
                f"In chunked mode the residual already observes the full chunk, so lookahead_k "
                f"must equal chunk_size ({self.chunk_size}) or be left null."
            )

        # Per-rollout base-policy chunk cache (single-step path only).
        # idx == chunk_size is the "needs refill" sentinel for the first call.
        self._base_chunk_cache: Optional[np.ndarray] = None
        self._chunk_step_idx = self.chunk_size

        # Extracting RL policy config params.
        self.policy_type = self.cfg.policy.type
        self.policy_impedance_mode = self.cfg.policy.impedance_mode
        self.policy_smooth_gain_lambda = self.cfg.policy.smooth_gain_lambda
        self.shape_rewards = self.cfg.policy.shape_rewards
        self.residual_mag_schedule = self.cfg.policy.residual_mag_schedule
        self.residual_mag = self.cfg.policy.residual_mag
        self.gains_mag = self.cfg.policy.gains_mag

        # Train-time proprio sensor noise on RL actor/critic inputs only.
        # Base policy is read via sample_base_policy() and is not affected.
        obs_noise_cfg = self.cfg.policy.get("obs_noise", {}) or {}
        self.obs_noise_enabled = bool(obs_noise_cfg.get("enabled", False))
        self.obs_noise_eef_pos_std = float(obs_noise_cfg.get("eef_pos_std", 0.0))
        self.obs_noise_eef_quat_rad_std = float(obs_noise_cfg.get("eef_quat_rad_std", 0.0))
        self.obs_noise_gripper_qpos_std = float(obs_noise_cfg.get("gripper_qpos_std", 0.0))


        # TODO: Handling jumpstart logic; a bit hacky right now, eventually need to clean up.
        # If curriculum, need to keep track of current model performance.
        # NOTE: this needs to be done in init and not setup model to properly load saved curriculum info.
        if "jumpstart" in self.cfg.policy and self.cfg.policy.jumpstart is not False:
            self.jumpstart = self.cfg.policy.jumpstart
            self.jumpstart_n = self.cfg.policy.jumpstart_n
            self.jumpstart_beta = self.cfg.policy.jumpstart_beta
            self.jumpstart_ma = self.cfg.policy.jumpstart_ma
        else:
            self.jumpstart = None
            self.jumpstart_n = 10
            self.jumpstart_beta = 0.05
            self.jumpstart_ma = 3
        
        if "jumpstart_buffer" in self.cfg.policy:
            self.jumpstart_buffer = self.cfg.policy.jumpstart_buffer
        else:
            self.jumpstart_buffer = True
        
        # Load base-policy stats whenever a path is configured. Consumers of
        # base_avg_horizon / base_avg_success_rate (jumpstart curriculum/random
        # branches) gate their own use; base_p95_ee_force feeds the optional
        # force-penalty fallback and is independent of jumpstart.
        base_stats_path = self.cfg.base_stats_path
        if not os.path.exists(base_stats_path):
            warnings.warn(f"Base stats file not found at {base_stats_path}; base policy performance stats will be unavailable.")
            self.base_avg_horizon = None
            self.base_avg_success_rate = None
            self.base_p95_ee_force = None
        else:
            ext = os.path.splitext(base_stats_path)[1]
            if ext == ".txt":
                arr = np.loadtxt(base_stats_path, dtype=float)
                base_stats = {
                    "avg_horizon": float(arr[0]),
                    "avg_success_rate": float(arr[1]),
                }
            elif ext in (".yaml", ".yml"):
                base_stats = OmegaConf.to_container(OmegaConf.load(base_stats_path))
            else:
                raise ValueError(f"Unsupported base_stats extension: {ext}")
            self.base_avg_horizon = base_stats.get("avg_horizon", None)
            self.base_avg_success_rate = base_stats.get("avg_success_rate", None)
            self.base_p95_ee_force = base_stats.get("force_p95", None)

        # Extracting controller params.
        self.controller_configs = self.cfg.controller
        assert self.policy_impedance_mode == self.controller_configs.impedance_mode, "Controller impedance mode in cfg.controller must match cfg.policy.impedance_mode"

        # Inline super()._setup_model() so buffer and policy get different obs spaces.
        # Buffer stores the full env obs (all keys incl. base policy's images).
        # Policy (actor/critic) only sees the RL subset.
        self._setup_lr_schedule()
        # Critic/value-net schedule: falls back to actor schedule when not specified.
        if self._critic_lr_input is None:
            self.critic_lr_schedule = self.lr_schedule
        else:
            self.critic_lr_schedule = get_schedule_fn(self._critic_lr_input)
        self.set_random_seed(self.seed)

        # Setting replay buffer.
        # base_action_dim = lookahead_k * per-slot dim. Per-slot dim equals action_space.shape[0]
        # divided by the residual's effective chunk (1 for single-step, chunk_size otherwise),
        # so this is impedance-aware without re-reading the controller cfg here.
        effective_rl_chunk = 1 if self.rl_single_step else self.chunk_size
        slot_dim = self.action_space.shape[0] // effective_rl_chunk
        base_action_dim = self.lookahead_k * slot_dim
        self.replay_buffer_class = DictFastBuffer
        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            base_action_dim=base_action_dim,
        )

        # Build RL-only observation space from policy.observation_meta.
        # This is a subset of the full env obs space — actor/critic only see these keys.
        # Under variable impedance, controller_state is auto-emitted by the env wrapper
        # and auto-included here (symmetric with how the residual action space grows by +2).
        rl_obs_meta = self.cfg.policy.observation_meta
        rl_low_dim = list(rl_obs_meta.low_dim_keys)
        if self.policy_impedance_mode == "variable":
            rl_low_dim.append("controller_state")
        rl_obs_spaces = {}
        for key in rl_low_dim + list(rl_obs_meta.get("image_keys", [])):
            rl_obs_spaces[key] = self.observation_space[key]
        self.rl_observation_space = spaces.Dict(rl_obs_spaces)

        if len(rl_obs_spaces) > 1 or any(len(s.shape) > 1 for s in rl_obs_spaces.values()):
            features_extractor_class = partial(CombinedExtractor, normalized_image=True)
        else:
            features_extractor_class = FlattenExtractor
        self.policy_class = partial(
            ResidualSACPolicy,
            features_extractor_class=features_extractor_class,
            policy_type=self.policy_type,
            impedance_mode=self.policy_impedance_mode,
            residual_mag=self.residual_mag,
            gains_mag=self.gains_mag,
            chunk_size=self.chunk_size,
            diffusion_act_dim=self.action_dim,
            rl_single_step=self.rl_single_step,
            lookahead_k=self.lookahead_k,
            critic_use_base_action=self.critic_use_base_action,
        )
        
        self.policy = self.policy_class(
            self.rl_observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self._convert_train_freq()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
        

        # Overwrite model policy to create base-conditioned fast policy.
        if self.policy_type not in ["residual", "residual_scale", "residual_force", "residual_scale2", "residual_force2"]:
            raise NotImplementedError("Only 'residual' policy type is implemented for FAST.")

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor # fast policy
        self.critic = self.policy.critic # fast critic
        self.critic_target = self.policy.critic_target # fast critic target, using reward shaping

    def _update_learning_rate(self, optimizers) -> None:
        # Override: critic optimizer follows self.critic_lr_schedule, everything
        # else follows self.lr_schedule. When no separate critic_lr is configured,
        # critic_lr_schedule IS lr_schedule, so behavior matches the SB3 default.
        progress = self._current_progress_remaining
        actor_lr = self.lr_schedule(progress)
        critic_lr = self.critic_lr_schedule(progress)
        self.logger.record("train/learning_rate", actor_lr)
        if critic_lr != actor_lr:
            self.logger.record("train/critic_learning_rate", critic_lr)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            lr = critic_lr if optimizer is self.critic.optimizer else actor_lr
            update_learning_rate(optimizer, lr)

    def _critic_action(self, actions: th.Tensor, base_action: th.Tensor) -> th.Tensor:
        """Concatenate the lookahead base_action onto the residual action for critic input.

        No-op when critic_use_base_action is False, preserving the legacy critic shape and
        the loadability of pre-flag checkpoints. When True, the critic's first layer was
        widened in policies.make_critic to accept the cat.
        """
        if not self.critic_use_base_action:
            return actions
        return th.cat([actions, base_action], dim=-1)

    def _filter_rl_obs(self, obs, add_noise: bool = True):
        # Buffer stores full env obs; actor/critic only know the RL subset.
        if isinstance(obs, dict):
            filtered = {k: obs[k] for k in self.rl_observation_space.spaces}
            if self.obs_noise_enabled and add_noise:
                if self.obs_noise_eef_pos_std > 0.0 and "robot0_eef_pos" in filtered:
                    filtered["robot0_eef_pos"] = _add_gaussian(
                        filtered["robot0_eef_pos"], self.obs_noise_eef_pos_std
                    )
                if self.obs_noise_gripper_qpos_std > 0.0 and "robot0_gripper_qpos" in filtered:
                    filtered["robot0_gripper_qpos"] = _add_gaussian(
                        filtered["robot0_gripper_qpos"], self.obs_noise_gripper_qpos_std
                    )
                if self.obs_noise_eef_quat_rad_std > 0.0 and "robot0_eef_quat" in filtered:
                    filtered["robot0_eef_quat"] = _quat_tangent_noise(
                        filtered["robot0_eef_quat"], self.obs_noise_eef_quat_rad_std
                    )
            return filtered
        return obs

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        
        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        final_actor_losses, smooth_gain_losses = [], []

        if self.actor_gradient_steps < 0:
            actor_gradient_idx = np.linspace(0, gradient_steps-1, gradient_steps, dtype=int)
        else:
            actor_gradient_idx = np.linspace(int(gradient_steps / self.actor_gradient_steps) - 1, gradient_steps-1, self.actor_gradient_steps, dtype=int)

        # gradient steps should be set by utd.
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sample state
            action_dict = self.get_combined_log_prob(replay_data.observations, base_action=replay_data.base_actions)
            # actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            # log_prob = log_prob.reshape(-1, 1)
            actions_pi = action_dict['final_action']
            log_prob = action_dict['log_prob'].reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                # next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_action_dict = self.get_combined_log_prob(replay_data.next_observations, base_action=replay_data.next_base_actions)
                next_actions = th.tensor(self.policy.unscale_action(next_action_dict['final_action'].cpu().numpy())).to(self.device)
                next_log_prob = next_action_dict['log_prob']
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(
                    self._filter_rl_obs(replay_data.next_observations),
                    self._critic_action(next_actions, replay_data.next_base_actions),
                ), dim=1)
                if self.min_q_heads is not None:
                    # REDQ-style target: min over a random subset of size min_q_heads.
                    n_critics = next_q_values.shape[1]
                    m = min(int(self.min_q_heads), n_critics)
                    idx = th.randperm(n_critics, device=next_q_values.device)[:m]
                    next_q_values = th.min(next_q_values.index_select(1, idx), dim=1, keepdim=True).values
                else:
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    if self.critic_backup_combine_type == 'min':
                        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    elif self.critic_backup_combine_type == 'mean':
                        next_q_values = th.mean(next_q_values, dim=1, keepdim=True)
                # add entropy term
                if self.critic_entropy_bonus:
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = self.get_rewards(replay_data) + (1 - replay_data.dones) * self.gamma * next_q_values
                # target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
			# using action from the replay buffer
            current_q_values = self.critic(
                self._filter_rl_obs(replay_data.observations),
                self._critic_action(replay_data.actions, replay_data.base_actions),
            )

            # compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor) #  for type checker
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_clip_norm is not None:
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
            self.critic.optimizer.step()

            if gradient_step in actor_gradient_idx:
                # Compute actor loss
				# Alternative: actor_loss = th.mean(log_prob - qf1_pi)
				# Min over all critic networks
                q_values_pi = th.cat(self.critic(
                    self._filter_rl_obs(replay_data.observations),
                    self._critic_action(actions_pi, replay_data.base_actions),
                ), dim=1)
                if self.policy_gradient_type == 'ensemble_mean':
                    min_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == 'min':
                    min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == 'mean':
                    min_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # TODO: MAYBE APPLY SCHEDULE FOR SMOOTH GAIN LAMBDA?
                if self.policy_smooth_gain_lambda > 0.0 and self.policy_impedance_mode != "fixed":
                    smooth_gain_loss = self.smooth_gain_loss(actions_pi, replay_data.observations, self.policy_smooth_gain_lambda)
                    smooth_gain_losses.append(smooth_gain_loss.item())

                    # Debugging gradients; only for first gradient step.
                    if self.cfg.debug_gradients and gradient_step == 0:
                        # Backup original actor loss for logging
                        self.actor.optimizer.zero_grad()
                        actor_loss.backward(retain_graph=True)
                        grad_norm_actor = th.zeros((), device=self.device)
                        for p in self.actor.parameters():
                            if p.grad is not None:
                                grad_norm_actor += p.grad.norm().item() ** 2
                        self.logger.record("debug/actor_grad_norm", grad_norm_actor)
                        # Backup smoothness gain loss for logging
                        self.actor.optimizer.zero_grad()
                        smooth_gain_loss.backward(retain_graph=True)
                        grad_norm_smooth = th.zeros((), device=self.device)
                        for p in self.actor.parameters():
                            if p.grad is not None:
                                grad_norm_smooth += p.grad.norm().item() ** 2
                        grad_norm_smooth = th.sqrt(grad_norm_smooth)
                        self.logger.record("debug/smooth_gain_grad_norm", grad_norm_smooth)
                    else:
                        # Optimize the actor with combined loss.
                        final_actor_loss = actor_loss + smooth_gain_loss
                        self.actor.optimizer.zero_grad()
                        final_actor_loss.backward()
                    if self.grad_clip_norm is not None:
                        th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                    self.actor.optimizer.step()
                else:
                    # Optimize the actor with actor loss only.
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    if self.grad_clip_norm is not None:
                        th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                    self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        if len(smooth_gain_losses) > 0:
            self.logger.record("train/smooth_gain_loss", np.mean(smooth_gain_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfFAST,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfFAST:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "actor", "critic", "critic_target", "diffusion_policy",
            "_random_thresholds", "_prev_dones",
        ] # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        # Standard SAC save parameters.
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def load_replay_buffer(self, path, truncate_last_traj: bool = True) -> None:
        """Load buffer and hard-error if its base_action width disagrees with current cfg.

        Without this guard, a lookahead_k mismatch between the saved buffer and current
        cfg surfaces as an opaque numpy broadcast error inside replay_buffer.add() partway
        into the first rollout. The buffer can't be resized in-place (would silently
        truncate or zero-pad cached base intent), so we tell the user to start fresh.
        """
        super().load_replay_buffer(path, truncate_last_traj=truncate_last_traj)
        loaded_dim = int(self.replay_buffer.base_actions.shape[-1])
        effective_rl_chunk = 1 if self.rl_single_step else self.chunk_size
        slot_dim = self.action_space.shape[0] // effective_rl_chunk
        cfg_dim = self.lookahead_k * slot_dim
        if loaded_dim != cfg_dim:
            raise ValueError(
                f"Replay buffer base_action width ({loaded_dim}) does not match the current cfg "
                f"({cfg_dim} = lookahead_k={self.lookahead_k} * slot_dim={slot_dim}). The buffer "
                f"was trained under a different lookahead_k and cannot be resumed in-place. Start "
                f"a fresh run instead, or revert lookahead_k to match the saved checkpoint."
            )
        # Sync attribute on buffers pickled before base_action_dim existed.
        if not hasattr(self.replay_buffer, "base_action_dim"):
            self.replay_buffer.base_action_dim = loaded_dim

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
        mask: Optional[np.ndarray] = None,
        base_action: Optional[np.ndarray] = None,
    ) -> None:
        """
        Overridden to cache base-policy actions in the replay buffer. Mirrors SB3's
        terminal_observation substitution, then calls pi0 once on the (substituted)
        next_obs so that bootstrap at timeouts uses pi0 of the true final obs.
        """
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        if self.rl_single_step:
            # Force a refill on any done so the next-state base action is conditioned
            # on the post-reset obs. Mid-chunk this also re-anchors all envs' chunks.
            # Returns a (B, lookahead_k * act_dim_eff) window matching the obs shape
            # the residual will see at replay time.
            next_base_action = self.peek_lookahead_actions(
                next_obs, self.lookahead_k, force_refill=bool(np.any(dones))
            )
        else:
            next_base_action = self.sample_base_policy(next_obs, return_numpy=True)

        if mask is not None:
            replay_buffer.add(
                self._last_original_obs,
                next_obs,
                buffer_action,
                reward_,
                dones,
                infos,
                mask=mask,
                base_action=base_action,
                next_base_action=next_base_action,
            )
        else:
            replay_buffer.add(
                self._last_original_obs,  # type: ignore[arg-type]
                next_obs,  # type: ignore[arg-type]
                buffer_action,
                reward_,
                dones,
                infos,
                base_action=base_action,
                next_base_action=next_base_action,
            )

        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _sample_action(
            self,
            learning_starts: int,
            action_noise: Optional[ActionNoise] = None,
            n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
		Sample an action according to the exploration policy.
		This is either done by sampling the probability distribution of the policy,
		or sampling a random action (from a uniform distribution over the action space)
		or by adding noise to the deterministic output.

		:param action_noise: Action noise that will be used for exploration
			Required for deterministic policy (e.g. TD3). This can also be used
			in addition to the stochastic policy for SAC.
		:param learning_starts: Number of steps before learning for the warm-up phase.
		:param n_envs:
		:return: action to take in the environment
			and scaled action that will be stored in the replay buffer.
			The two differs when the action space is not normalized (bounds are not [-1, 1]).
		"""
        # Select action randomly or according to policy.
        random_action_dict = {}
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase.
            random_action_dict['n_envs'] = n_envs
            if action_noise is not None:
                random_action_dict['action_noise'] = action_noise

        # Handling jumpstart logic.
        if self.jumpstart == "random":
            # Per-env handoff thresholds are sampled per-episode in collect_rollouts
            # (see self._random_thresholds). Each env uses base policy while its step
            # count is below its sampled threshold, residual policy after.
            env_step_count = np.array(self.env.env_method("get_step_count"))
            use_base = env_step_count < self._random_thresholds
            residual_mag = min(self.num_timesteps / self.residual_mag_schedule, 1.0)
        elif self.jumpstart == "curriculum":
            env_step_count = np.array(self.env.env_method("get_step_count"))
            horizon_threshold = (1.0 - (self.jumpstart_stage) / self.jumpstart_n) * self.base_avg_horizon * self.chunk_size
            use_base = env_step_count < horizon_threshold
            # TODO: for now, automatically use full residual mag for jsrl
            residual_mag = 1.0
        else:
            # Setting residual scale based on residual scale schedule.
            use_base = np.zeros(n_envs, dtype=bool)
            residual_mag = min(self.num_timesteps / self.residual_mag_schedule, 1.0)

        action_dict = self.get_combined_action(
            observation=self._last_obs,
            deterministic=False,
            random_action_dict=random_action_dict,
            residual_mag=residual_mag,
        )
        final_action = action_dict['final_action']
        base_action = action_dict['base_action']
        # Single-slot executable base action (== base_action under the chunked path,
        # but a single slot vs the lookahead-window base_action under single-step).
        base_action_exec = action_dict['base_action_exec']

        # Checking jump-start logic.
        if self.jumpstart in ("random", "curriculum"):
            final_action = np.where(use_base[:, None], base_action_exec, final_action)

        buffer_action = final_action # Buffer action is the final combined action.
        return final_action, buffer_action, use_base, base_action
        
    def predict_diffused(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        sample_base: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        Returns a dict so callers can opt into auxiliary outputs (e.g. the
        lookahead base_action window for eval-time Q-stitching) without forcing
        signature changes on every consumer when new fields are added. Always
        present: 'final_action'. Usually present: 'scaled_action', 'base_action'.

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: dict with 'final_action' (env action), 'scaled_action' (residual
            output, may be None), 'base_action' (lookahead window of base policy
            slots used to condition the residual this step).
        """
        if self.jumpstart == "curriculum":
            # TODO: for now, automatically use full residual mag for jsrl
            residual_mag = 1.0
        else:
            # Setting residual scale based on residual scale schedule.
            residual_mag = min(self.num_timesteps / self.residual_mag_schedule, 1.0)

        action_dict = self.get_combined_action(
            observation=observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            sample_base=sample_base,
            random_action_dict={},
            residual_mag=residual_mag,
        )
        return {
            'final_action': action_dict['final_action'],
            'scaled_action': action_dict.get('scaled_action', None),
            'base_action': action_dict.get('base_action', None),
        }

    def get_combined_log_prob(
        self,
        observation: th.Tensor,
        base_action: Optional[th.Tensor] = None,
    ) -> dict[str, th.Tensor]:
        """
        Helper function to sample actions and log probabilities - needs to preserve computation graph
        for gradients, so returns tensors. If `base_action` is provided, skips the pi0 call.
        """
        return_dict = {}
        # First, sample action from base policy (unless a cached one was provided).
        # In single-step lookahead mode the cached `base_action` is a flattened
        # window of `lookahead_k` slots; the executed slot is the first one.
        if base_action is None:
            base_action = self.sample_base_policy(observation, return_numpy=False)
        if self.rl_single_step and self.lookahead_k > 1:
            exec_dim = base_action.shape[-1] // self.lookahead_k
            base_action_exec = base_action[..., :exec_dim]
        else:
            base_action_exec = base_action

        # Sample action from fast policy.
        full_obs = {"obs": self._filter_rl_obs(observation), "base_action": base_action}
        scaled_action, log_prob = self.policy.actor.action_log_prob(full_obs)

        # Combine base action and fast policy action.
        if isinstance(self.policy, ResidualSACPolicy):
            # Setting residual scale based on residual scale schedule.
            residual_mag = min(self.num_timesteps / self.residual_mag_schedule, 1.0)
            final_action_dict = self.policy.get_final_action(
                scaled_action, base_action_exec, residual_mag, use_numpy=False
            )
            scaled_action = final_action_dict["scaled_action"]
            final_action = final_action_dict["final_action"]
        else:
            raise NotImplementedError("Only ResidualSACPolicy is implemented for FAST.")

        # Add fast action and final action to return dict.
        return_dict['scaled_action'] = scaled_action
        return_dict['final_action'] = final_action
        return_dict['log_prob'] = log_prob
        return return_dict

    def get_combined_action(
            self,
            observation: Union[np.ndarray, dict[str, np.ndarray]],
            state: Optional[tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            sample_base: bool = False,
            random_action_dict: dict = {},
            residual_mag: float = 1.0,
    ) -> dict[str, np.ndarray]:
        return_dict = {}
        # Sample action from base policy. Single-step path peeks a lookahead
        # window of `lookahead_k` slots to feed the residual obs (peek doesn't
        # advance the cache pointer), then advances by exactly one slot for the
        # action that gets executed this step. Chunked path returns the full
        # chunk flat. `base_action` is the obs/buffer view; `base_action_exec`
        # is the single slot fed to get_final_action below.
        if self.rl_single_step:
            base_action = self.peek_lookahead_actions(observation, self.lookahead_k)
            base_action_exec = self.get_next_base_action(observation)
        else:
            base_action = self.sample_base_policy(observation, return_numpy=True)
            base_action_exec = base_action
        return_dict['base_action'] = base_action
        return_dict['base_action_exec'] = base_action_exec

        # If sample_base is True, return only the base action as the final action.
        if sample_base:
            return_dict['final_action'] = base_action_exec
            return return_dict

        # Sample action from fast policy.
        if random_action_dict:
            # If provided, sample random action for exploration.
            action_noise = random_action_dict.get('action_noise', None)
            unscaled_action = np.array([self.actor.action_space.sample() for _ in range(random_action_dict['n_envs'])])
            if isinstance(self.action_space, spaces.Box):
                scaled_action = self.actor.scale_action(unscaled_action)

                # Add noise to the action (improve exploration)
                if action_noise is not None:
                    scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
                
                unscaled_action = self.actor.unscale_action(scaled_action)
            else:
                scaled_action = unscaled_action
        else:
            # Use predict() otherwise; this unsquashes, so re-scale if needed.
            # Eval rollouts (deterministic=True) read clean obs; training rollouts noised.
            full_obs = {
                "obs": self._filter_rl_obs(observation, add_noise=not deterministic),
                "base_action": base_action,
            }
            unscaled_action, _ = self.predict(full_obs, state, episode_start, deterministic)
            if isinstance(self.actor.action_space, spaces.Box):
                scaled_action = self.actor.scale_action(unscaled_action)
            else:
                scaled_action = unscaled_action

        # Combine base action and fast policy action. In single-step lookahead
        # mode, scaled_action is one slot wide; combine with the executed slot
        # only (not the lookahead window).
        if isinstance(self.policy, ResidualSACPolicy):
            final_action_dict = self.policy.get_final_action(
                scaled_action, base_action_exec, residual_mag, use_numpy=True
            )
            scaled_action = final_action_dict["scaled_action"]
            final_action = final_action_dict["final_action"]
            return_dict["predict_second_return"] = final_action_dict.get("predict_second_return", None)
        else:
            raise NotImplementedError("Only ResidualSACPolicy is implemented for FAST.")

        # Add fast action and final action to return dict.
        return_dict['scaled_action'] = scaled_action
        return_dict['unscaled_action'] = unscaled_action
        return_dict['final_action'] = final_action

        return return_dict
    
    def sample_base_policy(
        self,
        observation: Union[np.ndarray, th.Tensor, dict[str, Union[np.ndarray, th.Tensor]]],
        return_numpy: bool,
        full: bool = False,
    ) -> Union[np.ndarray, th.Tensor]:
        """
        Sample action from base policy. Passes per-key obs dict to the base
        policy wrapper, which selects its own subset via base_obs_meta.

        Args:
            full: If False (default), returns the chunked (B, chunk_size * act_dim_eff)
                output — the chunked-residual obs path. If True, returns the full
                (B, prediction_horizon * act_dim_eff) prediction without truncation —
                used by the single-step lookahead cache.
        """
        if isinstance(observation, dict):
            # Convert each key to numpy for base policy wrapper.
            base_observation = {}
            for key, val in observation.items():
                if isinstance(val, th.Tensor):
                    base_observation[key] = val.cpu().numpy().astype(np.float32)
                else:
                    base_observation[key] = np.asarray(val, dtype=np.float32)
        else:
            if isinstance(observation, np.ndarray):
                base_observation = observation.astype(np.float32)
            else:
                base_observation = observation.cpu().numpy().astype(np.float32)

        if full:
            base_action = self.diffusion_policy.predict_full_chunk(base_observation, return_numpy=return_numpy)
            horizon = self.prediction_horizon
        else:
            base_action = self.diffusion_policy(base_observation, return_numpy=return_numpy)
            horizon = self.chunk_size
        base_action = base_action.reshape(-1, horizon * self.action_dim)
        if self.policy_impedance_mode != "fixed":
            base_action = self.augment_controller_action(base_action, is_numpy=return_numpy, horizon=horizon)

        if return_numpy:
            if not isinstance(base_action, np.ndarray):
                base_action = base_action.cpu().numpy()
            return base_action.astype(np.float32)
        else:
            if isinstance(base_action, np.ndarray):
                base_action = th.from_numpy(base_action)
            return base_action.to(device=self.device, dtype=th.float32)
        
    def reset_base_cache(self):
        """Invalidate the single-step base-policy chunk cache.

        Call between contexts that independently drive the base policy (e.g.
        between training rollouts and eval episodes, where batch shapes and obs
        distributions differ). After reset, the next `get_next_base_action` /
        `peek_next_base_action` call will refill.
        """
        self._base_chunk_cache = None
        self._chunk_step_idx = self.chunk_size

    def _refill_base_chunk_cache(self, observation):
        """Query the base policy on `observation` and overwrite the cache for all envs.

        The cache holds the FULL prediction_horizon, not just chunk_size. Execution
        still consumes only the first chunk_size slots (after which a fresh refill
        triggers); the extra slots `[chunk_size : prediction_horizon)` provide
        lookahead context for the residual policy via `peek_lookahead_actions`.

        Resets the step idx to 0. Used by both `get_next_base_action` and
        `peek_next_base_action` / `peek_lookahead_actions` — refill events are shared
        across all envs (lockstep replan), so cache semantics match whether the
        trigger is a chunk boundary or a mid-chunk episode reset in any env.
        """
        chunk = self.sample_base_policy(observation, return_numpy=True, full=True)
        # (n_envs, prediction_horizon * act_dim_eff) -> (n_envs, prediction_horizon, act_dim_eff)
        act_dim_eff = chunk.shape[-1] // self.prediction_horizon
        self._base_chunk_cache = chunk.reshape(-1, self.prediction_horizon, act_dim_eff)
        self._chunk_step_idx = 0

    def get_next_base_action(self, observation):
        """Single-step path: return the current chunk slot and advance idx.

        Refills the cache from `observation` when the previous iteration exhausted
        the chunk or flagged a forced refill (e.g. via `peek_next_base_action` on a
        done transition). Refill cadence is keyed off chunk_size (execution horizon),
        not prediction_horizon (cache depth).
        """
        if self._chunk_step_idx >= self.chunk_size:
            self._refill_base_chunk_cache(observation)
        action = self._base_chunk_cache[:, self._chunk_step_idx, :].copy()
        self._chunk_step_idx += 1
        return action

    def peek_next_base_action(self, observation, force_refill: bool = False):
        """Single-step path: return the *upcoming* chunk slot without advancing idx.

        Used by `_store_transition` for `next_base_action`. If `force_refill` (any
        env reset this step) or the chunk is exhausted, eagerly refills from
        `observation` so the returned action is conditioned on the new episode's
        obs and the next rollout iteration is a cache hit.
        """
        if force_refill or self._chunk_step_idx >= self.chunk_size:
            self._refill_base_chunk_cache(observation)
        return self._base_chunk_cache[:, self._chunk_step_idx, :].copy()

    def peek_lookahead_actions(self, observation, k: int, force_refill: bool = False):
        """Single-step path: return the next `k` cached base actions, flattened.

        Returns shape (n_envs, k * act_dim_eff). Used to feed the residual policy
        a window of upcoming base intent (instead of just one slot). Shares the
        same refill semantics as `peek_next_base_action`. Validation in _setup_model
        guarantees `chunk_size + k - 1 <= prediction_horizon`, so the slice is
        always in-bounds without forcing extra base-policy inference.
        """
        if force_refill or self._chunk_step_idx >= self.chunk_size:
            self._refill_base_chunk_cache(observation)
        end = self._chunk_step_idx + k
        window = self._base_chunk_cache[:, self._chunk_step_idx:end, :].copy()
        n_envs = window.shape[0]
        return window.reshape(n_envs, -1)

    def get_rewards(self, replay_data):
        """
        For clarity, should type replay_data argument: dict or ReplayBufferSamples?
        """
        if self.shape_rewards:
            rewards = replay_data.rewards + self.get_shaped_rewards(replay_data.actions, replay_data.observations)
        else:
            rewards = replay_data.rewards
        return rewards

    def get_shaped_rewards(self, action, obs):
        return -self.cfg.policy.time_penalty
    
    def smooth_gain_loss(self, actions, observations, smooth_gain_lambda):
        assert self.control_obs, "smooth_gain_loss currently only supports control_obs=True."
        bs = actions.shape[0]

        action_gains = actions.reshape(bs, self.chunk_size, -1)[..., :2]  # Assuming first two dims are gains.
        current_gains = observations[..., -2:]  # Assuming last two dims of obs are current gains.

        gain_smoothness_penalty = th.cat([
            current_gains.unsqueeze(1) - 2 * action_gains[:, [0], :] + action_gains[:, [1], :], # inter-chunk penalty
            action_gains[:, 0:-2, :] - 2 * action_gains[:, 1:-1, :] + action_gains[:, 2:, :], # intra-chunk penalty
        ], dim=1)
        gain_smoothness_penalty = gain_smoothness_penalty.pow(2).mean() * smooth_gain_lambda
        return gain_smoothness_penalty

    def augment_controller_action(self, action, is_numpy: bool = True, horizon: Optional[int] = None):
        """
        Helper function to augment action with impedance parameters based on controller config. This does nothing if
        impedance mode is fixed, so it can always be called. The input action should be a flattened time-major
        chunk of length `horizon * self.action_dim`.

        Args:
            horizon: Number of timesteps in the chunk. Defaults to self.chunk_size for backwards compat.
                Pass self.prediction_horizon when augmenting a full base-policy chunk for the lookahead cache.

        Note: the default control params are assumed to be 0 in the normalized/scaled action space.
        """
        if self.policy_impedance_mode == "fixed":
            return action

        if horizon is None:
            horizon = self.chunk_size

        # Variable impedance: prepend damping + stiffness scalar channels (both zero in the
        # normalized action space, so the base policy contributes neutral gains).
        n_actions = action.shape[0]
        action = action.reshape(n_actions, horizon, self.action_dim)
        if is_numpy:
            damping_action = np.zeros((n_actions, horizon, 1), dtype=np.float32)
            stiffness_action = np.zeros((n_actions, horizon, 1), dtype=np.float32)
            action = np.concatenate([damping_action, stiffness_action, action], axis=-1)
        else:
            damping_action = th.zeros((n_actions, horizon, 1), device=action.device, dtype=th.float32)
            stiffness_action = th.zeros((n_actions, horizon, 1), device=action.device, dtype=th.float32)
            action = th.cat([damping_action, stiffness_action, action], dim=-1)
        action = action.reshape(n_actions, horizon * (self.action_dim + 2))

        return action

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: Union[int, Tuple[int, str]],
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: int = 4,
    ):
        """
        Overwriting collect_rollouts to handle jumpstart-specific logic.
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        # Fresh rollout batch — drop any chunk cache populated by a previous eval
        # (which may have run at a different n_envs) or a previous rollout that
        # ended mid-chunk.
        if self.rl_single_step:
            self.reset_base_cache()

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # For jumpstart="random": lazy-seed per-env handoff thresholds on first
            # iteration, and resample for any env that finished an episode on the
            # previous step. Ephemeral state (not saved), rebuilt on resume.
            if self.jumpstart == "random":
                hi = self.base_avg_horizon * self.chunk_size
                if self._random_thresholds is None:
                    self._random_thresholds = np.random.uniform(0, hi, size=env.num_envs)
                    self._prev_dones = np.zeros(env.num_envs, dtype=bool)
                elif self._prev_dones.any():
                    reset_mask = self._prev_dones
                    self._random_thresholds[reset_mask] = np.random.uniform(
                        0, hi, size=int(reset_mask.sum())
                    )

            # Select action randomly or according to policy
            actions, buffer_actions, use_base, base_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # If jumpstart_buffer is True, don't mask rollout transitions.
            buffer_mask = ~use_base if not self.jumpstart_buffer else None

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # Cache dones so the next iteration can resample thresholds for envs
            # whose episodes just ended (get_step_count() resets to 0 on env reset).
            if self.jumpstart == "random":
                self._prev_dones = dones.copy()

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            # TODO: only do this if we are in learning stage
            self._store_transition(
                replay_buffer, 
                buffer_actions, 
                new_obs, 
                rewards, 
                dones, 
                infos, 
                mask=buffer_mask, 
                base_action=base_actions, # next_base_actions handled in _store_transition()
            )  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

