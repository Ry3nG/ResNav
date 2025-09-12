"""Wrappers: Lidar frame stacking for VecEnv.

Provides a VecEnv wrapper `LidarFrameStackVec` that stacks lidar observations
across the last K timesteps with flattened output.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


class LidarFrameStackVec(VecEnvWrapper):
    """Stack lidar observations over time for a VecEnv.
    
    Specialized wrapper that only stacks the "lidar" key from Dict observations,
    with fixed configuration: flattened output and latest-first ordering.
    
    Args:
        venv: The vectorized environment to wrap
        k: Number of frames to stack (default: 4)
    """

    def __init__(self, venv: VecEnv, k: int = 4) -> None:
        super().__init__(venv)
        self.k = int(k)
        assert isinstance(self.observation_space, spaces.Dict), "LidarFrameStackVec requires Dict observation space"
        assert "lidar" in self.observation_space.spaces, "LidarFrameStackVec requires 'lidar' key in observation space"
        
        lidar_space = self.observation_space.spaces["lidar"]
        assert isinstance(lidar_space, spaces.Box) and len(lidar_space.shape) == 1, "Lidar space must be 1D Box"

        # Build new observation space
        new_spaces: Dict[str, spaces.Space] = {}
        for kname, space in self.observation_space.spaces.items():
            if kname == "lidar":
                low = np.repeat(space.low, self.k)
                high = np.repeat(space.high, self.k)
                new_spaces[kname] = spaces.Box(low=low, high=high, dtype=space.dtype)
            else:
                new_spaces[kname] = space
        self.observation_space = spaces.Dict(new_spaces)

        # Frame buffers per env for lidar
        self._lidar_buffers: List[Deque[np.ndarray]] = [
            deque(maxlen=self.k) for _ in range(self.num_envs)
        ]

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.venv.reset()
        assert isinstance(obs, dict)
        
        # Clear and seed buffers with initial lidar observations
        for e in range(self.num_envs):
            self._lidar_buffers[e].clear()
            val = np.array(obs["lidar"][e], copy=True)
            for _ in range(self.k):
                self._lidar_buffers[e].append(val.copy())
        
        return self._stack_obs(obs)

    def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        obs, rewards, dones, infos = self.venv.step_wait()
        assert isinstance(obs, dict)
        
        # Update lidar buffers with current observations
        for e in range(self.num_envs):
            cur = np.array(obs["lidar"][e], copy=True)
            self._lidar_buffers[e].append(cur)
        
        # Build stacked observations
        stacked_obs = self._stack_obs(obs)
        
        # Handle episode termination: reseed buffers and update terminal observations
        for e in range(self.num_envs):
            if dones[e]:
                last = np.array(obs["lidar"][e], copy=True)
                self._lidar_buffers[e].clear()
                for _ in range(self.k):
                    self._lidar_buffers[e].append(last.copy())
                
                # Replace terminal observation if provided
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
            if kname == "lidar":
                stacked = []
                for e in range(self.num_envs):
                    # Get frames in latest-first order
                    frames = list(reversed(list(self._lidar_buffers[e])))
                    mat = np.stack(frames, axis=0)  # (k, dim)
                    mat = mat.reshape(-1)  # Flatten to (k*dim,)
                    stacked.append(mat)
                out[kname] = np.stack(stacked, axis=0)
            else:
                out[kname] = arr
        return out