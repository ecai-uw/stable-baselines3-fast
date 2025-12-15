from typing import Any, Optional, Union

import torch as th
from gymnasium import spaces
import numpy as np
from torch import nn

from stable_baselines3.common.policies import BaseModel, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy, LOG_STD_MAX, LOG_STD_MIN

class BaseCriticValue(BaseModel):
    """
    Critic and value network class for a given base policy. This avoids the initialization of
    an unused policy network, and handles FAST-specific input processing (i.e. filtering velocity-based
    observation features).
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int],
        features_extractor_class: type[BaseFeaturesExtractor],
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        post_linear_modules: Optional[list[type[nn.Module]]] = None,
        non_vel_indices: Optional[list[int]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.non_vel_indices = non_vel_indices

        # Creating value kwargs.
        self.value_kwargs = {
            "observation_space": observation_space,
            "action_space": action_space,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "normalize_images": normalize_images,
            "post_linear_modules": post_linear_modules,
        }

        # Create critic and critic target kwargs.
        self.critic_kwargs = self.value_kwargs.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "share_features_extractor": share_features_extractor,
            }
        )
        self.share_features_extractor = share_features_extractor

        # Some sanity checks.
        assert share_features_extractor is False, "Sharing features extractor for base critic is not supported."

        # Build.
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Updating all network kwargs with features extractor.
        value_kwargs = self._update_features_extractor(self.value_kwargs)
        critic_kwargs = self._update_features_extractor(self.critic_kwargs)
        critic_target_kwargs = self._update_features_extractor(self.critic_kwargs)

        # Create networks.
        self.value_net = ContinuousValue(**value_kwargs).to(self.device)
        self.critic = ContinuousCritic(**critic_kwargs).to(self.device)
        self.critic_target = ContinuousCritic(**critic_target_kwargs).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Create optimizers.
        self.value_net.optimizer = self.optimizer_class(
            self.value_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode.
        self.critic_target.set_training_mode(False)

    def forward(self, obs: th.Tensor, action: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        raise NotImplementedError()

    def filter_obs(self, obs: th.Tensor) -> th.Tensor:
        if self.non_vel_indices is not None:
            return obs[:, self.non_vel_indices]
        else:
            return obs

    def forward_q(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        return self.critic(self.filter_obs(obs), action)

    def forward_qt(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        return self.critic_target(self.filter_obs(obs), action)

    def forward_v(self, obs: th.Tensor) -> th.Tensor:
        return self.value_net(self.filter_obs(obs))


class ContinuousValue(BaseModel):
    """
    Simple implementation for a continuous state-based value function.
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        post_linear_modules: Optional[list[type[nn.Module]]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        # Create value network.
        self.value_net = nn.Sequential(
            *create_mlp(features_dim, 1, net_arch, activation_fn, post_linear_modules=post_linear_modules)
        )
        self.add_module("value_net", self.value_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        value = self.value_net(features)
        return value


class ResidualActor(Actor):

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        post_linear_modules: Optional[list[type[nn.Module]]] = None,
        standard_gauss_init: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
            post_linear_modules,
            standard_gauss_init,
        )

    def get_action_dist_params(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """
        Overwriting self.get_action_dist_params() to separately encode features for observation and base
        action. This is written to work with parent (Actor) implementation of forward() and action_log_prob().
        """
        # this needs to extract obs and base action from the obs tensor dict
        obs_tensor = obs['obs']
        base_action_tensor = obs['base_action']
        features = self.extract_features(obs_tensor, self.features_extractor)
        latent_pi = self.latent_pi(th.cat([features, base_action_tensor], dim=1))
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        if not self.standard_gauss_init:
            log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, dict()


MAX_ACTION_SCALE = 10.0 # hardcoded for now - bounds of action scaling for residual_scale policy


class ResidualSACPolicy(SACPolicy):
    """
    Policy class wrapper that implements residual action prediction for SACPolicy.
    """

    actor: ResidualActor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        post_linear_modules: Optional[list[type[nn.Module]]] = None,
        standard_gauss_init: bool = False,
        policy_type: str = "residual",
        chunk_size: int = 1,
        act_dim: int = 1,
    ):
        self.policy_type = policy_type
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            post_linear_modules,
            standard_gauss_init,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Overwriting self._build() to allow actor architecture customization based on policy type.
        """
        # Updating actor kwargs based on policy type.
        if self.policy_type == "residual":
            # For residual policy, default action space is fine.
            pass
        elif self.policy_type in ["residual_scale", "residual_force", "residual_scale2", "residual_force2"]:
            # For residual + scale policy, add extra dimension to action space.
            actor_action_space = self.actor_kwargs["action_space"]
            self.actor_kwargs["action_space"] = spaces.Box(
                low=np.concatenate([actor_action_space.low, np.array([-1.0] * self.chunk_size)], axis=0),
                high=np.concatenate([actor_action_space.high, np.array([1.0] * self.chunk_size)], axis=0),
                dtype=actor_action_space.dtype,
            )
        else:
            raise ValueError("policy_type must be in ['residual', 'residual_scale', 'residual_force'].")
        # Calling parent _build() with updated actor kwargs.
        super()._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        """
        Overwriting self.make_actor() to allow actor to condition on base action by adding additional feature
        dimensions - leaves other parts (Critic, etc.) unchanged.
        """
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["features_dim"] += self.action_space.shape[0]
        return ResidualActor(**actor_kwargs).to(self.device)

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overwriting self.predict() to allow proper processing of environment observation and base action.
        Observation should contain keys 'obs' and 'base_action'.
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        # Grabbing residual actor inputs.
        obs = observation['obs']
        base_action = observation['base_action']

        obs_tensor, vectorized_env = self.obs_to_tensor(obs)
        obs_tensor_dict = {
            'obs': obs_tensor,
            'base_action': th.as_tensor(base_action, device=self.device)
        }

        with th.no_grad():
            actions = self._predict(obs_tensor_dict, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.actor.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.actor.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.actor.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.actor.action_space.low, self.actor.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]

    def get_final_action(
        self,
        scaled_action: np.ndarray,
        base_action: np.ndarray,
        residual_mag: float,
        use_numpy: bool = True,
    ):
        # Scale action (clip residuals based on residual magnitude).
        if self.policy_type == "residual":
            # NOTE: this assumes action space is normalized to [-1, 1]
            if use_numpy:
                scaled_action = np.clip(scaled_action, -residual_mag, residual_mag)
                final_action = np.clip(base_action + scaled_action, -1.0, 1.0)
            else:
                scaled_action = th.clamp(scaled_action, -residual_mag, residual_mag)
                final_action = th.clamp(base_action + scaled_action, -1.0, 1.0)
            return {
                "scaled_action": scaled_action,
                "final_action": final_action,
            }

        elif self.policy_type in ["residual_scale", "residual_force"]:
            # NOTE: this assumes unscaled action space is centered at 0...
            # ... which is NOT true for orientation....
            # but for now we are only scaling delta xyz
            # action_min = np.array([-1, -1, -1, -0.5288763, -1, -0.92734307, -1])
            # action_max = np.array([1, 1, 1, 0.62915826, 1, 0.42358947, 1 ])
            # zero_action = -action_min / (action_max - action_min) * 2 - 1
            # delta_action = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            # NOTE: THIS ASSUMES DELTA EEF ACTION SPACE
            if use_numpy:
                # Clamp residuals and scales with the same bounds. TODO: separate?
                scaled_action = np.clip(scaled_action, -residual_mag, residual_mag)
                pred_residual = scaled_action[:, :self.chunk_size * self.act_dim]
                pred_scale = scaled_action[:, -self.chunk_size:]
                if self.policy_type == "residual_force":
                    # Force scale to speed up.
                    pred_scale = np.maximum(pred_scale, 0.1)

                # Scaling final delta action.
                final_action = (base_action + pred_residual).reshape(-1, self.chunk_size, self.act_dim)
                pred_scale = pred_scale.reshape(-1, self.chunk_size, 1)
                final_action[:, :, 0:3] *= np.power(MAX_ACTION_SCALE, pred_scale) # only scale position
                final_action = final_action.reshape(-1, self.chunk_size * self.act_dim)
                final_action = np.clip(final_action, -1.0, 1.0)
            else:
                # Clamp residuals and scales with the same bounds. TODO: separate?
                scaled_action = th.clamp(scaled_action, -residual_mag, residual_mag)
                pred_residual = scaled_action[:, :self.chunk_size * self.act_dim]
                pred_scale = scaled_action[:, -self.chunk_size:]
                if self.policy_type == "residual_force":
                    # Force scale to speed up.
                    pred_scale = th.maximum(pred_scale, th.tensor(0.1, device=pred_scale.device))

                # Scaling finsal delta action.
                final_action = (base_action + pred_residual).view(-1, self.chunk_size, self.act_dim)
                pred_scale = pred_scale.view(-1, self.chunk_size, 1)
                final_action[:, :, 0:3] *= th.pow(MAX_ACTION_SCALE, pred_scale) # only scale position
                final_action = final_action.view(-1, self.chunk_size * self.act_dim)
                final_action = th.clamp(final_action, -1.0, 1.0)
            
            return {
                "scaled_action": scaled_action,
                "final_action": final_action,
                "predict_second_return": pred_scale,
            }
        
        elif self.policy_type in ["residual_scale2", "residual_force2"]:
            # Scale first, then apply residual.
            if use_numpy:
                # Clamp residuals and scales with the same bounds. TODO: separate?
                scaled_action = np.clip(scaled_action, -residual_mag, residual_mag)
                pred_residual = scaled_action[:, :self.chunk_size * self.act_dim]
                pred_scale = scaled_action[:, -self.chunk_size:]
                if self.policy_type == "residual_force2":
                    # Force scale to speed up.
                    pred_scale = np.maximum(pred_scale, 0.1)
                
                # Scaling final delta action.
                final_action = base_action.reshape(-1, self.chunk_size, self.act_dim)
                pred_scale = pred_scale.reshape(-1, self.chunk_size, 1)
                final_action[:, :, 0:3] *= np.power(MAX_ACTION_SCALE, pred_scale) # only scale position
                final_action = final_action.reshape(-1, self.chunk_size * self.act_dim)
                final_action = final_action + pred_residual
                final_action = np.clip(final_action, -1.0, 1.0)
            else:
                # Clamp residuals and scales with the same bounds. TODO: separate?
                scaled_action = th.clamp(scaled_action, -residual_mag, residual_mag)
                pred_residual = scaled_action[:, :self.chunk_size * self.act_dim]
                pred_scale = scaled_action[:, -self.chunk_size:]
                if self.policy_type == "residual_force2":
                    # Force scale to speed up.
                    pred_scale = th.maximum(pred_scale, th.tensor(0.1, device=pred_scale.device))
                
                # Scaling finsal delta action.
                final_action = base_action.view(-1, self.chunk_size, self.act_dim)
                pred_scale = pred_scale.view(-1, self.chunk_size, 1)
                final_action[:, :, 0:3] *= th.pow(MAX_ACTION_SCALE, pred_scale) # only scale position
                final_action = final_action.view(-1, self.chunk_size * self.act_dim)
                final_action = final_action + pred_residual
                final_action = th.clamp(final_action, -1.0, 1.0)
            return {
                "scaled_action": scaled_action,
                "final_action": final_action,
                "predict_second_return": pred_scale,
            }

        return scaled_action, final_action