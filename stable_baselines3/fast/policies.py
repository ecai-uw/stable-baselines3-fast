from typing import Any, Optional, Union

import torch as th
from gymnasium import spaces
import numpy as np
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, DiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy, LOG_STD_MAX, LOG_STD_MIN

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
    ):
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
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]