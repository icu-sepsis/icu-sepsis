import gym
import gymnasium
import numpy as np
from gym import ActionWrapper as GymActionWrapper
from gym import spaces as GymSpaces
from gymnasium import ActionWrapper, spaces


class FlattenActionWrapper(ActionWrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.action_levels = super().action_space.nvec
        self.action_space = spaces.Discrete(np.prod(self.action_levels))

    def action(self, action):
        return action // self.action_levels[1], action % self.action_levels[1]


class LegacyFlattenActionWrapper(GymActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_levels = super().action_space.nvec
        self.action_space = GymSpaces.Discrete(np.prod(self.action_levels))

    def action(self, action):
        return action // self.action_levels[1], action % self.action_levels[1]
