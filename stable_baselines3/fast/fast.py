from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.fast.policies import BaseCriticValue, ResidualSACPolicy

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from tqdm import tqdm
from functools import partial
import warnings

SelfFAST = TypeVar("SelfFAST", bound="FAST")

class FAST(OffPolicyAlgorithm):
    """
    FAST

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param base_kwargs: Arguments for the base (slow) critic.
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
	:param diffusion_act_dim: The action dimension for the diffusion policy (tuple of (action chunk length, action_dim))
	:param critic_backup_combine_type: How to combine the critics for the backup (min or mean)
    :param base_gamma: the discount factor for the base (slow) critic
    :param policy_action_condition: Whether to condition fast policy on base action
    :param shape_rewards: Whether to use PBRS from base policy values.
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
    critic_base: ContinuousCritic
    base_critic_value: BaseCriticValue

    def __init__(
		self,
		policy: Union[str, type[SACPolicy]],
		env: Union[GymEnv, str],
        base_kwargs: Optional[dict[str, Any]] = None,
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
		diffusion_act_dim=(1, 1),
        critic_backup_combine_type='min',
		base_gamma: float = 0.995,
        policy_action_condition: bool = False,
        shape_rewards: bool = False,
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
        self.diffusion_act_chunk = diffusion_act_dim[0]
        self.diffusion_act_dim = diffusion_act_dim[1]
        self.critic_backup_combine_type = critic_backup_combine_type
        self.base_gamma = base_gamma
        self.base_kwargs = base_kwargs
        # self.policy_type = policy_type
        self.policy_action_condition = policy_action_condition
        self.shape_rewards = shape_rewards


        if _init_setup_model:
            self._setup_model()
        
    def _setup_model(self) -> None:
        # Extracting base policy config params.
        self.policy_type = self.cfg.policy.type
        self.residual_mag_schedule = self.cfg.policy.residual_mag_schedule
        self.residual_mag = self.cfg.policy.residual_mag
        self.drop_velocity = self.cfg.policy.drop_velocity

        # Observation meta for processing observations.
        self.observation_meta = self.cfg.env.observation_meta

        # Sanity checks.
        assert self.base_gamma >= self.gamma, "base (slow) gamma should be larger than or equal to gamma"

        # Updating Policy and Actor clases.
        if self.policy_action_condition:
            self.policy_class = partial(
                ResidualSACPolicy, 
                policy_type=self.policy_type, 
                chunk_size=self.diffusion_act_chunk,
                act_dim=self.diffusion_act_dim,
            )
            # self.policy_class = ResidualSACPolicy # this will need to create the residual actor internally

        # Processing observation meta.
        if self.drop_velocity:
            self.non_vel_indices = []
            total_obs_size = 0
            for key, size in self.observation_meta.items():
                if 'qpos' not in key:
                    self.non_vel_indices.extend(list(range(total_obs_size, total_obs_size + size)))
                total_obs_size += size
            if len(self.non_vel_indices) == 0:
                raise ValueError("All observation indices are velocity indices.")
            if len(self.non_vel_indices) == total_obs_size:
                raise ValueError("drop_velocity is set but no velocity indices found; check observation_meta settings.")
        else:
            self.non_vel_indices = None
            
        super()._setup_model()
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
        # if self.policy_type != 'residual':
        if self.policy_type not in ["residual", "residual_scale", "residual_force"]:
            raise NotImplementedError("Only 'residual' policy type is implemented for FAST.")

        # Creating new base policy to get base critic.
        # TODO: In the long term, this logic is probably cleaner if it just overwrites the _setup_model of OffPolicyAlgorithm
        if self.drop_velocity:
            if not isinstance(self.observation_space, spaces.Box):
                raise ValueError("drop_velocity is currently only supported for Box observation spaces.")
            base_observation_space = spaces.Box(
                low=self.observation_space.low[self.non_vel_indices],
                high=self.observation_space.high[self.non_vel_indices],
                dtype=self.observation_space.dtype,
            )
        else:
            base_observation_space = self.observation_space

        # Creating base critic and base value net.
        self.base_kwargs.update({
            "observation_space": base_observation_space,
            "action_space": self.action_space,
            "lr_schedule": self.lr_schedule,
            "features_extractor_class": self.policy.features_extractor_class,
            "non_vel_indices": self.non_vel_indices,
        })
        self.base_critic_value = BaseCriticValue(**self.base_kwargs).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor # fast policy 
        self.critic = self.policy.critic # fast critic
        self.critic_target = self.policy.critic_target # fast critic target, using reward shaping

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

        if self.actor_gradient_steps < 0:
            actor_gradient_idx = np.linspace(0, gradient_steps-1, gradient_steps, dtype=int)
        else:
            actor_gradient_idx = np.linspace(int(gradient_steps / self.actor_gradient_steps) - 1, gradient_steps-1, self.actor_gradient_steps, dtype=int)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sample state
            action_dict = self.get_combined_log_prob(replay_data.observations)
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
                next_action_dict = self.get_combined_log_prob(replay_data.next_observations)
                next_actions = th.tensor(self.policy.unscale_action(next_action_dict['final_action'].cpu().numpy())).to(self.device)
                next_log_prob = next_action_dict['log_prob']
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
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
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

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
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                if self.critic_backup_combine_type == 'min':
                    min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == 'mean':
                    min_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
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
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def train_base_value(self, fqe_steps: int, vd_steps: int, batch_size: int = 64, vd_samples: int = 16) -> None:
        """
        Run Fitted Q Evaluation (FQE) to train the base (slow) critic, then distill
        into the base value function (VD). NOTE: this function does not unscale actions, but it shouldn't 
        matter, since we only use the distilled state value function.

        :param total_steps: Total number of gradient steps to take.
        :param batch_size: Batch size for each gradient update.
        """
        # Sanity check.
        if not self.shape_rewards:
            # TODO: this is really hacky, may need to clean up alter
            warnings.simplefilter("always")
            # ANSI escape codes for colors
            RED = "\033[91m"
            YELLOW = "\033[93m"
            RESET = "\033[0m"

            def colored_warning(message, category, filename, lineno, file=None, line=None):
                # Customize the warning message format
                print(f"{YELLOW}Warning: {message} ({category.__name__}) at {filename}:{lineno}{RESET}")

            # Override the default showwarning
            warnings.showwarning = colored_warning
            warnings.warn("FAST.train_base_value() called with shape_rewards=False; learned value will not be used for reward shaping.", UserWarning)
        
        # Switch to train mode.
        self.base_critic_value.critic.set_training_mode(True)
        # Create learning rate scheduler.
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.base_critic_value.critic.optimizer, T_max=fqe_steps)

        # for step in tqdm(range(total_steps)):
        with tqdm(range(fqe_steps)) as pbar:
            for step in pbar:
                # Sample replay buffer.
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                # observations, actions, next_observations, dones, rewards

                with th.no_grad():
                    # Sample actions from base diffusion policy.
                    noise = th.randn((batch_size, self.diffusion_act_chunk, self.diffusion_act_dim), device=self.device)
                    actions = self.diffusion_policy(replay_data.observations, noise, return_numpy=False)
                    actions = actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)

                    # Sample next actions from base diffusion policy.
                    next_noise = th.randn((batch_size, self.diffusion_act_chunk, self.diffusion_act_dim), device=self.device)
                    next_actions = self.diffusion_policy(replay_data.next_observations, next_noise, return_numpy=False)
                    next_actions = next_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)

                    # Compute next Q values using base critic target.
                    next_q_values = th.cat(self.base_critic_value.forward_qt(replay_data.next_observations, next_actions), dim=1)
                    if self.critic_backup_combine_type == 'min':
                        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    elif self.critic_backup_combine_type == 'mean':
                        next_q_values = th.mean(next_q_values, dim=1, keepdim=True)

                    # Compute TD target.
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.base_gamma * next_q_values

                # Compute current Q values, with actions in the buffer.
                current_q_values = self.base_critic_value.forward_q(replay_data.observations, replay_data.actions)

                # Compute critic loss.
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                pbar.set_postfix(critic_loss=critic_loss.item())

                # Optimize.
                self.base_critic_value.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.base_critic_value.critic.optimizer.step()
                lr_scheduler.step()

                # Update target networks.
                if step % self.target_update_interval == 0:
                    # NOTE: this skips updating running stats, since the critic is assumed to not use batch norm
                    polyak_update(self.base_critic_value.critic.parameters(), self.base_critic_value.critic_target.parameters(), self.tau)

        self.base_critic_value.critic.set_training_mode(False)

        # Now, distill base critic into base value function.
        self.base_critic_value.value_net.set_training_mode(True)
        value_lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.base_critic_value.value_net.optimizer, T_max=vd_steps)
        with tqdm(range(vd_steps)) as pbar:
            for step in pbar:
                # Sample replay buffer.
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

                with th.no_grad():
                    # Sample actions from base diffusion policy.
                    noise = th.randn((batch_size * vd_samples, self.diffusion_act_chunk, self.diffusion_act_dim), device=self.device)
                    obs = replay_data.observations.unsqueeze(1).expand(-1, vd_samples, -1).reshape(batch_size * vd_samples, -1)
                    actions = self.diffusion_policy(obs, noise, return_numpy=False)
                    actions = actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
                    
                    # Compute Q values using base critic; average across critics and across vd_samples.
                    mean_q_values = th.cat(self.base_critic_value.forward_q(obs, actions), dim=1).mean(dim=1, keepdim=True)
                    mean_q_values = mean_q_values.reshape(batch_size, vd_samples).mean(dim=1, keepdim=True)
                
                # Compute current V values.
                current_v_values = self.base_critic_value.forward_v(replay_data.observations)

                # Compute value loss.
                value_loss = F.mse_loss(current_v_values, mean_q_values)
                pbar.set_postfix(value_loss=value_loss.item())

                # Optimize.
                self.base_critic_value.value_net.optimizer.zero_grad()
                value_loss.backward()
                self.base_critic_value.value_net.optimizer.step()
                value_lr_scheduler.step()

        self.base_critic_value.value_net.set_training_mode(False)

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
        return super()._excluded_save_params() + ["actor", "critic", "critic_target", "diffusion_policy", "base_critic_value"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        # Standard SAC save parameters.
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]

        # FAST specific save parameters.
        state_dicts.extend(["base_critic_value"])
        state_dicts.extend([
            "base_critic_value.critic.optimizer",
            "base_critic_value.value_net.optimizer",
        ])
        return state_dicts, saved_pytorch_variables
    
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
        # Setting residual scale based on residual scale schedule.
        residual_mag = max(
            (self.num_timesteps / self.residual_mag_schedule) * self.residual_mag, self.residual_mag
        )
        action_dict = self.get_combined_action(
            observation=self._last_obs,
            deterministic=False,
            random_action_dict=random_action_dict,
            residual_mag=residual_mag,
        )
        final_action = action_dict['final_action']
        buffer_action = final_action # Buffer action is the final combined action.
        return final_action, buffer_action
        
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
        action_dict = self.get_combined_action(
            observation=observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            sample_base=sample_base,
            random_action_dict={},
            residual_mag=self.residual_mag,
        )
        return action_dict['final_action'], action_dict.get('predict_second_return', None)

    def get_combined_log_prob(
        self,
        observation: th.Tensor,        
    ) -> dict[str, th.Tensor]:
        """
        Helper function to sample actions and log probabilities - needs to preserve computation graph 
        for gradients, so returns tensors.
        """
        return_dict = {}
        # First, sample action from base policy.
        noise = th.randn((observation.shape[0], self.diffusion_act_chunk, self.diffusion_act_dim), device=self.device)
        base_action = self.diffusion_policy(observation, noise, return_numpy=False)
        base_action = base_action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)

        # Sample action from fast policy.
        full_obs = {"obs": observation, "base_action": base_action} if self.policy_action_condition else observation
        scaled_action, log_prob = self.policy.actor.action_log_prob(full_obs)
        # scaled_action = th.clamp(scaled_action, -self.residual_mag, self.residual_mag)
        
        # Combine base action and fast policy action.
        if isinstance(self.policy, ResidualSACPolicy):
            scaled_action, final_action = self.policy.get_final_action(
                scaled_action, base_action, self.residual_mag, use_numpy=False
            )
        else:
            raise NotImplementedError("Only ResidualSACPolicy is implemented for FAST.")
        # if self.policy_type == 'residual':
            # NOTE: this assumes action space is normalized to [-1, 1]
        #     final_action = th.clamp(base_action + scaled_action, -1, 1)
        # else:
        #     raise NotImplementedError("Only 'residual' policy type is implemented for FAST.")
        
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
            residual_mag: float = 0.0,
    ) -> dict[str, np.ndarray]:
        return_dict = {}
        # First, sample action from base policy.
        noise = th.randn((observation.shape[0], self.diffusion_act_chunk, self.diffusion_act_dim), device=self.device)
        base_action = self.diffusion_policy(
            th.as_tensor(observation, device=self.device, dtype=th.float32), 
            noise, 
            return_numpy=True,
        )
        base_action = base_action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
        return_dict['base_action'] = base_action

        # If sample_base is True, return only the base action as the final action.
        if sample_base:
            return_dict['final_action'] = base_action
            return return_dict

        # Sample action from fast policy.
        if random_action_dict:
            # If provided, sample random action for exploration.
            action_noise = random_action_dict.get('action_noise', None)
            # TODO: BUGFIX - THIS NEEDS TO SAPMLE FROM ACTOR ACTION SPACE
            unscaled_action = np.array([self.actor.action_space.sample() for _ in range(random_action_dict['n_envs'])])
            # unscaled_action = np.array([self.action_space.sample() for _ in range(random_action_dict['n_envs'])])
            if isinstance(self.action_space, spaces.Box):
                # scaled_action = self.policy.scale_action(unscaled_action)
                scaled_action = self.actor.scale_action(unscaled_action)

                # Add noise to the action (improve exploration)
                if action_noise is not None:
                    scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
                
                unscaled_action = self.actor.unscale_action(scaled_action)
                # unscaled_action = self.policy.unscale_action(scaled_action)
            else:
                scaled_action = unscaled_action
        else:
            # Use predict() otherwise -this unsquashes, so re-scale if needed.
            full_obs = {"obs": observation, "base_action": base_action} if self.policy_action_condition else observation
            unscaled_action, predict_second_return = self.predict(full_obs, state, episode_start, deterministic)
            if isinstance(self.actor.action_space, spaces.Box):
                scaled_action = self.actor.scale_action(unscaled_action)
            else:
                scaled_action = unscaled_action
            return_dict['predict_second_return'] = predict_second_return

        # Combine base action and fast policy action.
        if isinstance(self.policy, ResidualSACPolicy):
            scaled_action, final_action = self.policy.get_final_action(
                scaled_action, base_action, residual_mag, use_numpy=True
            )
        else:
            raise NotImplementedError("Only ResidualSACPolicy is implemented for FAST.")
        # if self.policy_type == 'residual':
        #     scaled_action = np.clip(scaled_action, -residual_mag, residual_mag)
        #     # NOTE: this assumes action space is normalized to [-1, 1]
        #     final_action = np.clip(base_action + scaled_action, -1, 1)
        # else:
        #     raise NotImplementedError("Only 'residual' policy type is implemented for FAST.")

        # Add fast action and final action to return dict.
        return_dict['scaled_action'] = scaled_action
        return_dict['unscaled_action'] = unscaled_action
        return_dict['final_action'] = final_action

        return return_dict
    
    def get_rewards(self, replay_data):
        """
        For clarity, should type replay_data argument: dict or ReplayBufferSamples?
        """
        if self.shape_rewards:
            rewards = replay_data.rewards + \
                  self.gamma * self.base_critic_value.forward_v(replay_data.next_observations).detach() - \
                  self.base_critic_value.forward_v(replay_data.observations).detach()
        else:
            rewards = replay_data.rewards
        return rewards
    
    def get_shaped_rewards(self, obs, next_obs):
        # TODO: get_rewards function should directly call this function
        return self.gamma * self.base_critic_value.forward_v(next_obs).detach() - \
                self.base_critic_value.forward_v(obs).detach()