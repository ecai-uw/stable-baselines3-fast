import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class FastBuffer(ReplayBuffer):
    """
    Modified replay buffer with per-environment buffers for optional jumpstart learning.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        # offline_mix_ratio: int = -1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)
        # Re-define self.pos and full to separately track positions for each environment.
        self.pos = np.zeros(self.n_envs, dtype=np.int32)
        self.full = np.zeros(self.n_envs, dtype=bool)

        # For now, raise error for non-Box observation and action spaces
        if not isinstance(self.observation_space, spaces.Box):
            raise NotImplementedError("FastBuffer only supports Box observation spaces")
        if not isinstance(self.action_space, spaces.Box):
            raise NotImplementedError("FastBuffer only supports Box action spaces")
        
        # For now, raise error if optimizing memory usage.
        if optimize_memory_usage:
            raise NotImplementedError("FastBuffer does not support memory optimization")
        
        # For now, raise error if offline mix ratio is positive.
        if self.offline_mix_ratio > 0:
            raise NotImplementedError("FastBuffer does not support offline mixing")
    
    def size(self) -> int:
        # Total number of transitions stored across all environments.
        size = np.sum(np.where(self.full, self.buffer_size, self.pos))
        return int(size)

    def reset(self) -> None:
        self.pos = np.zeros(self.n_envs, dtype=np.int32)
        self.full = np.zeros(self.n_envs, dtype=bool)
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Optional[Union[dict, list]] = None,
        mask: Optional[np.ndarray] = None,
    ):
        if mask is not None:
            buffer_index = self.pos[mask]
            env_index = np.arange(self.n_envs)[mask]
        else:
            mask = np.ones(self.n_envs, dtype=bool)
            buffer_index = self.pos
            env_index = mask
        
        self.observations[buffer_index, env_index] = np.array(obs[mask])
        self.next_observations[buffer_index, env_index] = np.array(next_obs[mask])
        self.actions[buffer_index, env_index] = np.array(action[mask])
        self.rewards[buffer_index, env_index] = np.array(reward[mask])
        self.dones[buffer_index, env_index] = np.array(done[mask])
        if self.handle_timeout_termination:
            self.timeouts[buffer_index, env_index] = np.array([info.get("TimeLimit.truncated", False) for info, m in zip(infos, mask) if m])
        
        self.pos[mask] += 1
        # Updating full flags and wrapping positions for each env buffer.
        for e in range(self.n_envs):
            if self.pos[e] == self.buffer_size:
                self.full[e] = True
                self.pos[e] = 0
    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Modified buffer sampling to uniformly weight all transitions across variably sized environment buffers.
        env_upper_bound = np.where(self.full, self.buffer_size, self.pos)
        env_prob = env_upper_bound / env_upper_bound.sum()
        env_indices = np.random.choice(np.arange(self.n_envs).astype(int), size=batch_size, p=env_prob)
        batch_inds = np.random.randint(low=0, high=env_upper_bound[env_indices], size=batch_size)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
    
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        raise ValueError("Should not be called for FastBuffer")
    
class DictFastBuffer(FastBuffer):
    """
    Dict version of FastBuffer for image-based observations.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        
        self.buffer_size = max(buffer_size // n_envs, 1)  # Buffer size per environment

        assert not optimize_memory_usage, "FastBuffer does not support memory optimization"
        self.optimizer_memory_usage = optimize_memory_usage
        
        # Re-define self.pos and full to separately track positions for each environment.
        self.pos = np.zeros(self.n_envs, dtype=np.int32)
        self.full = np.zeros(self.n_envs, dtype=bool)

        if not isinstance(self.observation_space, spaces.Dict):
            raise NotImplementedError("FastBuffer only supports Dict observation spaces")
        if not isinstance(self.action_space, spaces.Box):
            raise NotImplementedError("FastBuffer only supports Box action spaces")
        
        # For now, raise error if optimizing memory usage.
        if optimize_memory_usage:
            raise NotImplementedError("FastBuffer does not support memory optimization")
        
        # For now, raise error if offline mix ratio is positive.
        if self.offline_mix_ratio > 0:
            raise NotImplementedError("FastBuffer does not support offline mixing")
        
        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
        
        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if not optimize_memory_usage:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )
        
    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Optional[Union[dict, list]] = None,
        mask: Optional[np.ndarray] = None,
    ):
        if mask is not None:
            buffer_index = self.pos[mask]
            env_index = np.arange(self.n_envs)[mask]
        else:
            mask = np.ones(self.n_envs, dtype=bool)
            buffer_index = self.pos
            env_index = mask
        
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            self.observations[key][buffer_index, env_index] = np.array(obs[key][mask])
            self.next_observations[key][buffer_index, env_index] = np.array(next_obs[key][mask])
        self.actions[buffer_index, env_index] = np.array(action[mask])
        self.rewards[buffer_index, env_index] = np.array(reward[mask])
        self.dones[buffer_index, env_index] = np.array(done[mask])
        if self.handle_timeout_termination:
            self.timeouts[buffer_index, env_index] = np.array([info.get("TimeLimit.truncated", False) for info, m in zip(infos, mask) if m])
        
        self.pos[mask] += 1
        # Updating full flags and wrapping positions for each env buffer.
        for e in range(self.n_envs):
            if self.pos[e] == self.buffer_size:
                self.full[e] = True
                self.pos[e] = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Modified buffer sampling to uniformly weight all transitions across variably sized environment buffers.
        env_upper_bound = np.where(self.full, self.buffer_size, self.pos)
        env_prob = env_upper_bound / env_upper_bound.sum()
        env_indices = np.random.choice(np.arange(self.n_envs).astype(int), size=batch_size, p=env_prob)
        batch_inds = np.random.randint(low=0, high=env_upper_bound[env_indices], size=batch_size)

        obs = {key: self.observations[key][batch_inds, env_indices, ...] for key in self.observation_space.keys()}
        next_obs = {key: self.next_observations[key][batch_inds, env_indices, ...] for key in self.observation_space.keys()}
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)

        return DictReplayBufferSamples(
            observations={key: self.to_torch(o) for key, o in obs.items()},
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations={key: self.to_torch(no) for key, no in next_obs.items()},
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch((self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )