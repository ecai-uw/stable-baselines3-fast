import os
from copy import deepcopy
from typing import Any, ClassVar, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
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

        # TODO: clean up; abstract as many params as possible into this cfg
        self.cfg = cfg

        self.diffusion_policy = diffusion_policy
        # self.chunk_size = cfg.base_policy.chunk_size
        # self.action_dim = cfg.base_policy.action_dim
        self.critic_backup_combine_type = critic_backup_combine_type

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

        # Per-rollout base-policy chunk cache (single-step path only).
        # idx == chunk_size is the "needs refill" sentinel for the first call.
        self._base_chunk_cache: Optional[np.ndarray] = None
        self._chunk_step_idx = self.chunk_size

        # Extracting RL policy config params.
        self.policy_type = self.cfg.policy.type
        self.policy_impedance_mode = self.cfg.policy.impedance_mode
        self.policy_gains_only = self.cfg.policy.gains_only
        self.policy_smooth_gain_lambda = self.cfg.policy.smooth_gain_lambda
        self.shape_rewards = self.cfg.policy.shape_rewards
        self.residual_mag_schedule = self.cfg.policy.residual_mag_schedule
        self.residual_mag = self.cfg.policy.residual_mag
        self.gains_mag = self.cfg.policy.gains_mag
        
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
        
        if self.jumpstart in ("curriculum", "random"):
            base_stats_path = self.cfg.base_stats_path
            if "simple_reset" in self.cfg.env and self.cfg.env.simple_reset:
                base, ext = os.path.splitext(base_stats_path)
                base_stats_path = base + "_simple_reset" + ext
            # Load base policy stats.
            base_stats = np.loadtxt(base_stats_path, dtype=float)
            self.base_avg_horizon = base_stats[0]
            self.base_avg_success_rate = base_stats[1]
        else:
            self.base_avg_horizon = None
            self.base_avg_success_rate = None

        # Extracting controller params.
        self.controller_configs = self.cfg.controller
        assert self.policy_impedance_mode == self.controller_configs.impedance_mode, "Controller impedance mode in cfg.controller must match cfg.policy.impedance_mode"

        # Inline super()._setup_model() so buffer and policy get different obs spaces.
        # Buffer stores the full env obs (all keys incl. base policy's images).
        # Policy (actor/critic) only sees the RL subset.
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Setting replay buffer.
        self.replay_buffer_class = DictFastBuffer
        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
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
            gains_only=self.policy_gains_only,
            residual_mag=self.residual_mag,
            gains_mag=self.gains_mag,
            chunk_size=self.chunk_size,
            diffusion_act_dim=self.action_dim,
            rl_single_step=self.rl_single_step,
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

    def _filter_rl_obs(self, obs):
        # Buffer stores full env obs; actor/critic only know the RL subset.
        if isinstance(obs, dict):
            return {k: obs[k] for k in self.rl_observation_space.spaces}
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
                next_q_values = th.cat(self.critic_target(self._filter_rl_obs(replay_data.next_observations), next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                if self.critic_backup_combine_type == 'min':
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == 'mean':
                    next_q_values = th.mean(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = self.get_rewards(replay_data) + (1 - replay_data.dones) * self.gamma * next_q_values
                # target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
			# using action from the replay buffer
            current_q_values = self.critic(self._filter_rl_obs(replay_data.observations), replay_data.actions)

            # compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor) #  for type checker
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if gradient_step in actor_gradient_idx:
                # Compute actor loss
				# Alternative: actor_loss = th.mean(log_prob - qf1_pi)
				# Min over all critic networks
                q_values_pi = th.cat(self.critic(self._filter_rl_obs(replay_data.observations), actions_pi), dim=1)
                if self.critic_backup_combine_type == 'min':
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
                    self.actor.optimizer.step()
                else:
                    # Optimize the actor with actor loss only.
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
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
            next_base_action = self.peek_next_base_action(
                next_obs, force_refill=bool(np.any(dones))
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

        # Checking jump-start logic.
        if self.jumpstart in ("random", "curriculum"):
            final_action = np.where(use_base[:, None], base_action, final_action)

        buffer_action = final_action # Buffer action is the final combined action.
        return final_action, buffer_action, use_base, base_action
        
    def predict_diffused(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        sample_base: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
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
        return action_dict['final_action'], action_dict.get('scaled_action', None)

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
        if base_action is None:
            base_action = self.sample_base_policy(observation, return_numpy=False)

        # Sample action from fast policy.
        full_obs = {"obs": self._filter_rl_obs(observation), "base_action": base_action}
        scaled_action, log_prob = self.policy.actor.action_log_prob(full_obs)
        
        # Combine base action and fast policy action.
        if isinstance(self.policy, ResidualSACPolicy):
            # Setting residual scale based on residual scale schedule.
            residual_mag = min(self.num_timesteps / self.residual_mag_schedule, 1.0)
            final_action_dict = self.policy.get_final_action(
                scaled_action, base_action, residual_mag, use_numpy=False
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
        # First, sample action from base policy.
        # Single-step path: pop one action from the cached chunk (refilled as needed).
        # Chunked path: return the full chunk flattened to (B, chunk_size * act_dim_eff).
        if self.rl_single_step:
            base_action = self.get_next_base_action(observation)
        else:
            base_action = self.sample_base_policy(observation, return_numpy=True)
        return_dict['base_action'] = base_action

        # If sample_base is True, return only the base action as the final action.
        if sample_base:
            return_dict['final_action'] = base_action
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
            full_obs = {"obs": self._filter_rl_obs(observation), "base_action": base_action}
            unscaled_action, _ = self.predict(full_obs, state, episode_start, deterministic)
            if isinstance(self.actor.action_space, spaces.Box):
                scaled_action = self.actor.scale_action(unscaled_action)
            else:
                scaled_action = unscaled_action
            # return_dict['predict_second_return'] = predict_second_return

        # Combine base action and fast policy action.
        if isinstance(self.policy, ResidualSACPolicy):
            final_action_dict = self.policy.get_final_action(
                scaled_action, base_action, residual_mag, use_numpy=True
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
    ) -> Union[np.ndarray, th.Tensor]:
        """
        Sample action from base policy. Passes per-key obs dict to the base
        policy wrapper, which selects its own subset via base_obs_meta.
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

        base_action = self.diffusion_policy(base_observation, return_numpy=return_numpy)
        base_action = base_action.reshape(-1, self.chunk_size * self.action_dim)
        if self.policy_impedance_mode != "fixed":
            base_action = self.augment_controller_action(base_action, is_numpy=return_numpy)

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

        Resets the global step idx to 0. Used by both `get_next_base_action` and
        `peek_next_base_action` — refill events are shared across all envs (lockstep
        replan), so cache semantics match whether the trigger is a chunk boundary or
        a mid-chunk episode reset in any env.
        """
        chunk = self.sample_base_policy(observation, return_numpy=True)
        # (n_envs, chunk_size * act_dim_eff) -> (n_envs, chunk_size, act_dim_eff)
        act_dim_eff = chunk.shape[-1] // self.chunk_size
        self._base_chunk_cache = chunk.reshape(-1, self.chunk_size, act_dim_eff)
        self._chunk_step_idx = 0

    def get_next_base_action(self, observation):
        """Single-step path: return the current chunk slot and advance idx.

        Refills the cache from `observation` when the previous iteration exhausted
        the chunk or flagged a forced refill (e.g. via `peek_next_base_action` on a
        done transition).
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

    def augment_controller_action(self, action, is_numpy: bool = True):
        """
        Helper function to augment action with impedance parameters based on controller config. This does nothing if 
        impedance mode is fixed, so it can always be called. The input action should be an action-chunked, OSC 
        action (i.e. chunk_size * act_dim).

        Note: the default control params are assumed to be 0 in the normalized/scaled action space.
        """
        if self.policy_impedance_mode == "fixed":
            return action

        # Variable impedance: prepend damping + stiffness scalar channels (both zero in the
        # normalized action space, so the base policy contributes neutral gains).
        n_actions = action.shape[0]
        action = action.reshape(n_actions, self.chunk_size, self.action_dim)
        if is_numpy:
            damping_action = np.zeros((n_actions, self.chunk_size, 1), dtype=np.float32)
            stiffness_action = np.zeros((n_actions, self.chunk_size, 1), dtype=np.float32)
            action = np.concatenate([damping_action, stiffness_action, action], axis=-1)
        else:
            damping_action = th.zeros((n_actions, self.chunk_size, 1), device=action.device, dtype=th.float32)
            stiffness_action = th.zeros((n_actions, self.chunk_size, 1), device=action.device, dtype=th.float32)
            action = th.cat([damping_action, stiffness_action, action], dim=-1)
        action = action.reshape(n_actions, self.chunk_size * (self.action_dim + 2))

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

