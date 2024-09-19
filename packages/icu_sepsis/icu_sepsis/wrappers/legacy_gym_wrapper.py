"""Wrapper for the legacy gym environment."""

import gym
import gymnasium
import numpy as np
from gym import spaces
from gymnasium import spaces as gymn_spaces


class GymWrapper(gym.Env):
    """Wrapper for the legacy gym environment."""

    def __init__(self, env: gymnasium.Env, **kwargs):
        self._env = env

        self.action_space = self._convert_space(env.action_space)
        self.observation_space = self._convert_space(env.observation_space)

        self.metadata = env.metadata

        self._env.reset(seed=kwargs.get('seed'))

    def _convert_space(self, space: gymnasium.Space) -> gym.Space:
        if isinstance(space, gymn_spaces.Discrete):
            return spaces.Discrete(space.n)
        elif isinstance(space, gymn_spaces.MultiDiscrete):
            return spaces.MultiDiscrete(space.nvec)
        elif isinstance(space, gymn_spaces.Box):
            return spaces.Box(low=space.low, high=space.high,
                              shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gymn_spaces.MultiBinary):
            return spaces.MultiBinary(n=space.n)
        elif isinstance(space, gymn_spaces.Dict):
            return spaces.Dict({k: self._convert_space(v)
                                for k, v in space.spaces.items()})
        elif isinstance(space, gymn_spaces.Tuple):
            return spaces.Tuple([self._convert_space(s) for s in space.spaces])
        else:
            raise NotImplementedError(f'Unsupported space type: {type(space)}')

    def reset(self, *, seed: int = None, options: dict = None) -> int:
        obs, _ = self._env.reset(seed=seed, options=options)
        return obs

    def step(self, action: np.ndarray) -> tuple[int, float, bool, dict]:
        obs, rew, done, _, info = self._env.step(action)
        return obs, rew, done, info

    def render(self, mode: str = 'human'):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()

    # getattr from the wrapped environment
    def __getattr__(self, name):
        return getattr(self._env, name)
