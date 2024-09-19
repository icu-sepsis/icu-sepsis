"""The ICU Sepsis environment module."""

import logging
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..utils.io import MDPParameters
from ..utils import constants
from ..utils.constants import inadmissible_action_strategies as ias
from ..utils.exceptions import InadmissibleActionError


class ICUSepsisEnv(gym.Env):
    """## The ICU-Sepsis environment.

    The ICU-Sepsis environment is a discrete MDP that models the treatment of
    sepsis in an ICU. The state space consists of **716 states**, with the
    first 713 representing the state of a patient undergoing, and the
    last three representing death, survival, and s_inf respectively. The action
    space consists of **25 actions**, representing a combination of fluids and
    vasopressors to be administered to the patient at 5 different levels each.
    The episode ends when the agent reaches a terminal state or the maximum
    number of steps is reached.

    For more information, see: https://github.com/icu-sepsis/icu-sepsis
    """

    # Internal environment parameters
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    _tx_mat: np.ndarray
    _r_mat: np.ndarray
    _d_0: np.ndarray
    _expert_policy: np.ndarray
    _metadata: dict
    _admissible_actions: list[list[int]]
    _state_cluster_centers: np.ndarray
    _sofa_scores: np.ndarray

    def __init__(self, *, render_mode: str = None,
                 params: MDPParameters = None,
                 inadmissible_action_strategy: str = None, **kwargs):
        """Initializes the ICU Sepsis environment.

        Args:
            render_mode (str, optional):
                Rendering mode for an episode (Not implemented yet). Defaults
                to None.
            params (MDPParameters, optional):
                Custom parameters to load in the environment, used for
                development or debugging purposes. Defaults to None.
            inadmissible_action_strategy (str, optional):
                Strategy to handle inadmissible actions. Keeping it None will
                use the default strategy defined in the package constants.
                Defaults to None.
        """
        # Load provided environment parameters, or use default parameters
        if params is None:
            data_root = Path(__file__).parent.joinpath('assets')
            logging.debug('Loading data from %s', data_root)
            params = MDPParameters(data_root)
        params.load_to(self)

        # Set admissible actions as a list of sets for faster lookup
        self._admissible_action_sets = [
            set(admissible_actions) for admissible_actions in
            self._admissible_actions]

        # Set environment properties
        self._num_states: int = self._tx_mat.shape[0]
        self._num_actions: int = self._tx_mat.shape[1]
        self._gamma: float = kwargs.get('gamma', 1.00)
        self._action_levels: int = round(np.sqrt(self._num_actions))
        assert self._action_levels ** 2 == self._num_actions, \
            f'Invalid number of actions: {self._num_actions}'

        self.action_space = spaces.MultiDiscrete(
            [self._action_levels, self._action_levels])
        self.observation_space = spaces.Discrete(self._num_states, start=0)

        if inadmissible_action_strategy is None:
            self._inadmissible_action_strategy = \
                ias.DEFAULT_INADMISSIBLE_ACTION_STRATEGY
        else:
            assert inadmissible_action_strategy in [
                ias.MEAN, ias.TERMINATE, ias.RAISE_EXCEPTION], \
                'Invalid inadmissible action strategy.'
            self._inadmissible_action_strategy = inadmissible_action_strategy

        logging.info('Initialized ICU-Sepsis environment with gamma=%f',
                     self._gamma)

        # Set up rendering (Not implemented yet)
        if render_mode is not None:
            assert render_mode in self.metadata['render_modes'], \
                'Invalid render mode.'
            raise NotImplementedError('Rendering is not implemented yet.')

        # Reset the environment
        self._current_state: int | None = None
        self.reset(seed=kwargs.get('seed'))

    ###########################################################################
    # Helper functions
    ###########################################################################

    def _get_action_idx(self, action: np.ndarray) -> int:
        return action[0] * self.action_space.nvec[1] + action[1]

    def _is_terminal_state(self, state: int) -> bool:
        if (state < 0) or (state >= self._num_states):
            raise ValueError(f'Invalid state: {state}')
        return state in constants.STATES_TERMINAL

    def _is_admissible_action(self, state: int, action_idx: int) -> bool:
        return action_idx in self._admissible_action_sets[state]

    def _make_state_transition(self, p: np.ndarray):
        self._current_state = self.np_random.choice(self._num_states, p=p)

    def _take_action(self, state: int, action_idx: int):
        transition_prob = self._tx_mat[state, action_idx, :]

        # If the action is admissible, make the state transition
        if self._is_admissible_action(state, action_idx):
            self._make_state_transition(transition_prob)
            return

        # Handle inadmissible actions based on the strategy
        if self._inadmissible_action_strategy == ias.MEAN:
            self._make_state_transition(transition_prob)
        elif self._inadmissible_action_strategy == ias.TERMINATE:
            self._current_state = constants.STATE_DEATH
        elif self._inadmissible_action_strategy == ias.RAISE_EXCEPTION:
            raise InadmissibleActionError(
                f'Inadmissible action {action_idx} in state {state}')
        else:
            raise ValueError('Invalid inadmissible action strategy.')

    def _get_obs(self) -> int:
        return self._current_state

    def _get_info(self) -> dict:
        return {
            'admissible_actions': self._admissible_actions[
                self._current_state],
            'state_vector': self._state_cluster_centers[self._current_state],
            'sofa_score': self._sofa_scores[self._current_state]}

    @property
    def _terminated(self) -> bool:
        if self._current_state is None:
            return None
        return self._is_terminal_state(self._current_state)

    ###########################################################################
    # Gym environment methods
    ###########################################################################

    def reset(self, *, seed: int = None, options: dict = None
              ) -> tuple[int, dict]:
        super().reset(seed=seed, options=options)

        # Set the initial state
        self._make_state_transition(self._d_0)
        logging.debug(
            'Reset environment with initial state %s',
            self._current_state)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[int, float, bool, bool, dict]:
        assert not self._terminated, 'Cannot transition from a terminal state.'

        # Compute action index and save the current state
        action_idx = self._get_action_idx(action)
        prev_state = self._current_state

        # State transition based on action and inadmissible action strategy
        self._take_action(prev_state, action_idx)

        # Generate the reward based on the reward matrix
        reward = self._r_mat[prev_state, action_idx, self._current_state]

        return (self._get_obs(), reward, self._terminated,
                False, self._get_info())

    def render(self):
        raise NotImplementedError

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def num_states(self) -> int:
        """The number of states in the environment."""
        return self._num_states

    @property
    def num_actions(self) -> int:
        """The number of possible actions in the environment."""
        return self._num_actions

    @property
    def gamma(self) -> float:
        """The discount factor for future rewards."""
        return self._gamma

    @property
    def expert_policy(self) -> np.ndarray:
        """Cleaned version of the estimated expert policy used by clinicians
        to treat patients. The probabilities for actions taken less than `tao`
        times (given in the `env_metadata`) are set to 0, and the remaining
        probabilities are re-normalized accordingly to sum to 1."""
        return self._expert_policy

    @property
    def state_cluster_centers(self) -> np.ndarray:
        """Centroids of different clusters of states in the state space."""
        return self._state_cluster_centers

    @property
    def sofa_scores(self) -> np.ndarray:
        """Average Sequential Organ Failure Assessment (SOFA) scores of
        patients for each state in the state space."""
        return self._sofa_scores

    @property
    def env_metadata(self) -> dict:
        """Metadata regarding the environment creation process."""
        return self._metadata

    @property
    def dynamics(self) -> dict:
        """Dynamics of the environment, for debugging and analysis, should NOT
        be used for learning."""
        return {
            'tx_mat': self._tx_mat,
            'r_mat': self._r_mat,
            'd_0': self._d_0,
            'admissible_actions': self._admissible_actions}
