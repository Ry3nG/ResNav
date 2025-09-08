"""Wrappers: Dict frame stacking for VecEnv.

Provides a VecEnv wrapper `DictFrameStackVec` that stacks only selected keys
from a Dict observation across the last K timesteps. Supports flattened
stacking (default) or preserving a time dimension.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


class DictFrameStackVec(VecEnvWrapper):
    """Stack specified Dict obs keys over time for a VecEnv.

    - keys: iterable of dict keys to stack (e.g., ["lidar"]).
    - k: number of frames to stack.
    - flatten: if True, flattens stacked key to shape (orig_dim*k,).
      if False, shape is (k, orig_dim).
    - latest_first: if True, order is [t, t-1, ..., t-k+1], else chronological.
    """

    def __init__(
        self,
        venv: VecEnv,
        keys: Iterable[str],
        k: int = 4,
        flatten: bool = True,
        latest_first: bool = True,
    ) -> None:
        super().__init__(venv)
        self.keys = list(keys)
        self.k = int(k)
        self.flatten = bool(flatten)
        self.latest_first = bool(latest_first)
        assert isinstance(self.observation_space, spaces.Dict), "DictFrameStackVec requires Dict observation space"

        # Build new observation space
        new_spaces: Dict[str, spaces.Space] = {}
        for kname, space in self.observation_space.spaces.items():
            if kname in self.keys:
                assert isinstance(space, spaces.Box) and len(space.shape) == 1, "Stacked key must be 1D Box"
                n = space.shape[0]
                if self.flatten:
                    low = np.repeat(space.low, self.k)
                    high = np.repeat(space.high, self.k)
                    new_spaces[kname] = spaces.Box(low=low, high=high, dtype=space.dtype)
                else:
                    low = np.tile(space.low, self.k).reshape(self.k, n)
                    high = np.tile(space.high, self.k).reshape(self.k, n)
                    new_spaces[kname] = spaces.Box(low=low, high=high, dtype=space.dtype)
            else:
                new_spaces[kname] = space
        self.observation_space = spaces.Dict(new_spaces)

        # Frame buffers per env per key
        self._buffers: Dict[str, List[Deque[np.ndarray]]] = {
            kname: [deque(maxlen=self.k) for _ in range(self.num_envs)] for kname in self.keys
        }

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.venv.reset()
        assert isinstance(obs, dict)
        for kname in self.keys:
            for e in range(self.num_envs):
                self._buffers[kname][e].clear()
                # Seed with k copies of the initial observation for that env (batched)
                val = np.array(obs[kname][e], copy=True)
                for _ in range(self.k):
                    self._buffers[kname][e].append(val.copy())
        return self._stack_obs(obs)

    def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        obs, rewards, dones, infos = self.venv.step_wait()
        assert isinstance(obs, dict)
        # Update buffers with current obs for each env
        for e in range(self.num_envs):
            for kname in self.keys:
                cur = np.array(obs[kname][e], copy=True)
                self._buffers[kname][e].append(cur)
        # Build stacked obs from buffers
        stacked_obs = self._stack_obs(obs)
        # For done envs, reseed buffers with the (post-done) current obs and also
        # replace terminal_observation with the stacked version to match shapes
        for e in range(self.num_envs):
            if dones[e]:
                for kname in self.keys:
                    last = np.array(obs[kname][e], copy=True)
                    self._buffers[kname][e].clear()
                    for _ in range(self.k):
                        self._buffers[kname][e].append(last.copy())
                # Replace terminal observation if provided by inner env
                if "terminal_observation" in infos[e]:
                    term = infos[e]["terminal_observation"]
                    if isinstance(term, dict):
                        new_term: Dict[str, np.ndarray] = {}
                        for key in stacked_obs:
                            new_term[key] = stacked_obs[key][e]
                        infos[e]["terminal_observation"] = new_term
        return stacked_obs, rewards, dones, infos

    def _stack_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for kname, arr in obs.items():
            if kname not in self.keys:
                out[kname] = arr
                continue
            stacked = []
            for e in range(self.num_envs):
                frames = list(self._buffers[kname][e])
                if self.latest_first:
                    frames = frames[::-1]
                mat = np.stack(frames, axis=0)  # (k, dim)
                if self.flatten:
                    mat = mat.reshape(-1)
                stacked.append(mat)
            out[kname] = np.stack(stacked, axis=0)
        return out
