"""Helpers extracted from fast.py to keep that module focused on the algorithm."""
import os
from typing import Callable, Optional, Union

import numpy as np
import torch as th
from omegaconf import OmegaConf


def add_gaussian(x, std):
    """Additive iid Gaussian noise. Dispatches on np.ndarray vs torch.Tensor."""
    if isinstance(x, th.Tensor):
        return x + th.randn_like(x) * std
    return x + (np.random.standard_normal(x.shape) * std).astype(x.dtype, copy=False)


def quat_tangent_noise(quat, sigma_rad):
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


def augment_controller_action(action, impedance_mode: str, action_dim: int, horizon: int, is_numpy: bool = True):
    """Prepend variable-impedance gain channels (damping, stiffness) to a flat action chunk.

    No-op when impedance_mode == "fixed" (so callers can always invoke this). The input
    action is a flattened time-major chunk of length `horizon * action_dim`; output is
    `horizon * (action_dim + 2)`. Default control params are 0 in the normalized action
    space so the base policy contributes neutral gains.
    """
    if impedance_mode == "fixed":
        return action

    n_actions = action.shape[0]
    action = action.reshape(n_actions, horizon, action_dim)
    if is_numpy:
        damping = np.zeros((n_actions, horizon, 1), dtype=np.float32)
        stiffness = np.zeros((n_actions, horizon, 1), dtype=np.float32)
        action = np.concatenate([damping, stiffness, action], axis=-1)
    else:
        damping = th.zeros((n_actions, horizon, 1), device=action.device, dtype=th.float32)
        stiffness = th.zeros((n_actions, horizon, 1), device=action.device, dtype=th.float32)
        action = th.cat([damping, stiffness, action], dim=-1)
    return action.reshape(n_actions, horizon * (action_dim + 2))


def load_base_stats(path: str) -> Optional[dict]:
    """Load cached base-policy stats from .txt or .yaml. Returns None if file missing.

    .txt format: two whitespace-separated floats — `avg_horizon avg_success_rate`.
    .yaml format: a dict of named fields (avg_horizon, avg_success_rate, force_p95, ...).
    Caller decides defaults for absent fields and whether to warn on missing file.
    """
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1]
    if ext == ".txt":
        arr = np.loadtxt(path, dtype=float)
        return {"avg_horizon": float(arr[0]), "avg_success_rate": float(arr[1])}
    if ext in (".yaml", ".yml"):
        return OmegaConf.to_container(OmegaConf.load(path))
    raise ValueError(f"Unsupported base_stats extension: {ext}")


class BaseChunkCache:
    """Per-rollout cache of base-policy chunks for the single-step path.

    Holds the FULL prediction_horizon worth of base actions per refill; execution
    consumes only the first `chunk_size` slots before triggering a fresh refill.
    The extra `[chunk_size : prediction_horizon)` slots provide lookahead context
    for the residual policy via `peek_lookahead`.

    Refill events are shared across all envs (lockstep replan) regardless of whether
    the trigger is a chunk boundary or a mid-chunk forced refill (e.g. on episode
    reset). idx == chunk_size is the "needs refill" sentinel for the first call.
    """

    def __init__(
        self,
        chunk_size: int,
        prediction_horizon: int,
        sample_full: Callable[[Union[np.ndarray, dict]], np.ndarray],
    ):
        """
        Args:
            chunk_size: execution horizon — refill triggers when this many slots
                have been consumed since the last refill.
            prediction_horizon: depth of each refill — must satisfy
                `prediction_horizon >= chunk_size + max_lookahead_k - 1` for
                `peek_lookahead` to be in-bounds.
            sample_full: callable taking an observation and returning a flat
                (n_envs, prediction_horizon * act_dim_eff) base-policy chunk.
        """
        self.chunk_size = chunk_size
        self.prediction_horizon = prediction_horizon
        self._sample_full = sample_full
        self._cache: Optional[np.ndarray] = None
        self._idx = chunk_size  # sentinel: needs refill on first call

    def reset(self) -> None:
        """Invalidate the cache so the next call refills.

        Call between contexts that independently drive the base policy (e.g. between
        training rollouts and eval episodes, where batch shapes and obs distributions
        differ).
        """
        self._cache = None
        self._idx = self.chunk_size

    def _refill(self, observation) -> None:
        chunk = self._sample_full(observation)
        # (n_envs, prediction_horizon * act_dim_eff) -> (n_envs, prediction_horizon, act_dim_eff)
        act_dim_eff = chunk.shape[-1] // self.prediction_horizon
        self._cache = chunk.reshape(-1, self.prediction_horizon, act_dim_eff)
        self._idx = 0

    def get_next(self, observation) -> np.ndarray:
        """Return the current chunk slot and advance idx; refill if exhausted."""
        if self._idx >= self.chunk_size:
            self._refill(observation)
        action = self._cache[:, self._idx, :].copy()
        self._idx += 1
        return action

    def peek_next(self, observation, force_refill: bool = False) -> np.ndarray:
        """Return the *upcoming* chunk slot without advancing idx.

        If `force_refill` (any env reset this step) or the chunk is exhausted, eagerly
        refills from `observation` so the returned action is conditioned on the new
        episode's obs and the next iteration is a cache hit.
        """
        if force_refill or self._idx >= self.chunk_size:
            self._refill(observation)
        return self._cache[:, self._idx, :].copy()

    def peek_lookahead(self, observation, k: int, force_refill: bool = False) -> np.ndarray:
        """Return the next `k` cached base actions, flattened to (n_envs, k * act_dim_eff).

        Used to feed the residual policy a window of upcoming base intent. Shares the
        same refill semantics as `peek_next`. The constructor's prediction_horizon
        must be >= chunk_size + k - 1 for the slice to stay in-bounds without forcing
        an extra base-policy inference.
        """
        if force_refill or self._idx >= self.chunk_size:
            self._refill(observation)
        end = self._idx + k
        window = self._cache[:, self._idx:end, :].copy()
        return window.reshape(window.shape[0], -1)
