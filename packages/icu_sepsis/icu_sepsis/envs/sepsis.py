import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from icu_sepsis.utils.io import MDPParameters
from scipy.special import softmax


class ICUSepsisEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, *, render_mode: str = None,
                 params: MDPParameters = None, **kwargs):
        if params is None:
            # Load default params
            data_root = Path(__file__).parent.joinpath('assets')
            logging.debug('Loading data from %s', data_root)
            params = MDPParameters(data_root)

        self.tx_mat = params.tx_mat
        self.r_mat = params.r_mat
        self.d_0 = params.d_0
        self.data_policy = params.expert_policy
        self.allowed_actions = params.allowed_actions
        self.metadata = params.metadata
        self.state_cluster_centers = params.state_cluster_centers
        self.sofa_scores = params._sofa_scores

        self.terminal_states = self.tx_mat.shape[0]-3

        self.num_states = self.tx_mat.shape[0]
        self.num_actions = self.tx_mat.shape[1]

        self.action_levels = round(np.sqrt(self.num_actions))
        assert self.action_levels ** 2 == self.num_actions, \
            f'Invalid number of actions: {self.num_actions}'

        self.action_space = spaces.MultiDiscrete(
            [self.action_levels, self.action_levels])
        self.observation_space = spaces.Discrete(self.num_states, start=0)

        self.gamma: float = kwargs.get('gamma', 1.00)
        self._beta_obs_stoch: float = kwargs.get('beta_obs_stochasticity')
        self._beta_ac_stoch: float = kwargs.get('beta_ac_stochasticity')
        self._beta_treatment_delay: float = kwargs.get('beta_treatment_delay')

        logging.info('Initialized ICU-Sepsis environment with '
                     'gamma=%f, '
                     'beta_obs_stochasticity=%s, '
                     'beta_ac_stochasticity=%s, '
                     'beta_treatment_delay=%s',
                     self.gamma, self._beta_obs_stoch,
                     self._beta_ac_stoch, self._beta_treatment_delay)

        if self._beta_ac_stoch is not None:
            assert 0 <= self._beta_ac_stoch <= 1, \
                'Invalid action stochasticity coefficient.'
            eps = 1e-20
            self.ac_values = range(self.action_levels)
            indices = np.array(self.ac_values)
            ac_arr = np.array([
                (abs(i-indices) - self.action_levels) / (
                    np.log(abs(1-self._beta_ac_stoch)+eps) - eps)
                for i in self.ac_values]).T
            self._ac_probs = softmax(ac_arr, axis=1)

        # Reset the environment
        self.current_state: int | None = None
        self._terminated: bool | None = None
        self._treatment_started: bool = self._beta_treatment_delay is None
        self._ep_steps: int | None = None
        self.reset(seed=kwargs.get('seed'))

        # Set up rendering
        if render_mode is not None:
            assert render_mode in self.metadata['render_modes'], \
                'Invalid render mode.'
            raise NotImplementedError('Rendering is not implemented yet.')

    def _get_action_idx(self, action: np.ndarray) -> int:
        return action[0] * self.action_space.nvec[1] + action[1]

    def _get_info(self) -> dict:
        # return the list of allowed actions for the current state

        return {
            'allowed_actions': self.allowed_actions[self.current_state],
            'state_vector': self.state_cluster_centers[self.current_state],
            'sofa_score': self.sofa_scores[self.current_state]
        }

    def _get_obs(self) -> int:
        if (self.current_state >= self.num_states-3) or \
                (self._beta_obs_stoch is None) or \
                (self.np_random.uniform() > self._beta_obs_stoch):
            # if terminal state, or no stochasticity,
            # or stochasticity is not triggered
            return self.current_state

        # if stochasticity is triggered, sample from non-terminal states
        return self.np_random.choice(self.num_states-3)

    def _sample_action(self, action: np.ndarray) -> np.ndarray:
        if self._beta_ac_stoch is None:
            return action

        return np.array([self.np_random.choice(
            self.ac_values, p=self._ac_probs[n]) for n in action])

    def _attempt_treatment(self)->bool:
        if self._treatment_started:
            return True

        self._treatment_started = self.np_random.uniform() > self._beta_treatment_delay
        return self._treatment_started

    def reset(self, *,
              seed: int = None, options: dict = None) -> tuple[int, dict]:
        self._terminated = False
        super().reset(seed=seed)

        self._ep_steps = 0
        self._treatment_started = self._beta_treatment_delay is None
        self._attempt_treatment()

        # Set the initial state
        self.current_state = self.np_random.choice(self.num_states, p=self.d_0)
        logging.debug(
            'Reset environment with initial state %s',
            self.current_state)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[int, float, bool, bool, dict]:
        assert not self._terminated, 'Cannot transition from a terminal state.'

        # check if treatment has started
        if self._attempt_treatment():
            # sample actual action from decided action
            action = self._sample_action(action)
            action_idx = self._get_action_idx(action)
        else:
            # if treatment has not started, set action to 0
            action_idx = 0

        # Sample the next state using the transition matrix
        next_state = self.np_random.choice(
            self.num_states, 
            p = self.tx_mat[self.current_state, action_idx, :])

        # Generate the reward based on the reward matrix
        reward = self.r_mat[self.current_state, action_idx, next_state]
        self.current_state = next_state
        self._terminated = next_state >= self.terminal_states

        self._ep_steps += 1

        return (self._get_obs(), reward, self._terminated,
                False, self._get_info())

    def get_dynamics(self) -> dict:
        return {
            'tx_mat': self.tx_mat,
            'r_mat': self.r_mat,
            'd_0': self.d_0,
            'allowed_actions': self.allowed_actions
        }

    def get_expert_policy(self) -> np.ndarray:
        return self.data_policy

    def get_config(self) -> dict:
        return {
            'gamma': self.gamma,
            'beta_obs_stochasticity': self._beta_obs_stoch,
            'beta_ac_stochasticity': self._beta_ac_stoch,
            'beta_treatment_delay': self._beta_treatment_delay
        }

    def get_metadata(self) -> dict:
        return self.metadata
