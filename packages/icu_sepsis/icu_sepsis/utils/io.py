"""Helpers for loading and saving the parameters of the ICU-Sepsis
environment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class MDPParameters:
    """Container for the parameters of the ICU-Sepsis environment."""

    def __init__(self, data_dir: str | Path, /):
        data_dir = Path(data_dir)
        with open(data_dir.joinpath('metadata.json'), 'r') as f:
            self._metadata = json.load(f)
        try:
            data = np.load(data_dir.joinpath('dynamics.npz'))
            self._tx_mat = data['tx_mat']
            self._r_mat = data['r_mat']
            self._d_0 = data['d_0']
            self._expert_policy = data['expert_policy']
            self._state_cluster_centers = data['state_cluster_centers']
            self._sofa_scores = data['sofa_scores']
        except FileNotFoundError as e:
            self._state_cluster_centers = pd.read_csv(
                data_dir.joinpath('state_cluster_centers.csv'))
            # TODO: load data from csv files using the metadata
            raise NotImplementedError(
                'Loading from csv files is not implemented yet.') from e

        self._admissible_actions = self.read_admissible_actions(
            data_dir.joinpath('admissible_actions.txt'))

    @staticmethod
    def read_admissible_actions(infile_path: Path) -> list[list[int]]:
        """Reads the list of admissible actions from a file.

        Args:
            infile_path (Path):
                Path to the file containing admissible actions.

        Returns:
            list[list[int]]:
                List of admissible actions for each state.
        """
        n_acs = []
        admissible_actions = []
        with open(infile_path, 'r') as f:
            # Read number of admissible actions in the first line
            line_items = f.readline().strip().split()
            n_acs = [int(n) for n in line_items]

            # Read admissible actions for each state in the following lines
            for n in n_acs:
                line_items = f.readline().strip().split()
                assert len(line_items) == n, \
                    (f'Number of admissible actions ({n}) does not match the '
                     f'number of actions in the line ({len(line_items)}).')
                admissible_actions.append([int(ac) for ac in line_items])

        return admissible_actions

    @staticmethod
    def write_admissible_actions(admissible_actions: list[list[int]],
                                 outfile_path: Path):
        """Writes the list of admissible actions to a file.

        Args:
            admissible_actions (list[list[int]]):
                List of admissible actions for each state.

            outfile_path (Path):
                Path to the file to write admissible actions.
        """
        with open(outfile_path, 'w') as f:
            # write number of admissible actions in the first line
            for acs in admissible_actions:
                f.write(f'{len(acs)} ')
            f.write('\n')

            # write admissible actions for each state in the following lines
            for acs in admissible_actions:
                for ac in acs:
                    f.write(f'{ac} ')
                f.write('\n')

    @staticmethod
    def save_mat_as_csv(mat: np.ndarray, out_path: Path, /):
        """Saves a matrix as a CSV file.

        Args:
            mat (np.ndarray):
                Matrix to save.
            out_path (Path):
                Path to the output CSV file.
        """
        mat_flat = mat.reshape(-1, mat.shape[-1])
        pd.DataFrame(mat_flat).to_csv(out_path, index=False, header=False)

    @staticmethod
    def create(tx_mat: np.ndarray, r_mat: np.ndarray, d_0: np.ndarray,
               expert_policy: np.ndarray, admissible_actions: list[list[int]],
               state_cluster_centers: np.ndarray, sofa_scores: np.ndarray,
               metadata: dict) -> "MDPParameters":
        """Creates an MDPParameters object.

        Args:
            tx_mat (np.ndarray):
                Transition matrix.

            r_mat (np.ndarray):
                Reward matrix.

            d_0 (np.ndarray):
                Initial state distribution.

            expert_policy (np.ndarray):
                Expert policy.

            admissible_actions (list[list[int]]):
                List of admissible actions for each state.

            state_cluster_centers (np.ndarray):
                Centroids of the state clusters.

            sofa_scores (np.ndarray):
                SOFA scores for each state.

            metadata (dict):
                Metadata dictionary

        Returns:
            MDPParameters:
                Created MDPParameters object.
        """
        params = MDPParameters.__new__(MDPParameters)
        params._tx_mat = tx_mat
        params._r_mat = r_mat
        params._d_0 = d_0
        params._expert_policy = expert_policy
        params._admissible_actions = admissible_actions
        params._state_cluster_centers = state_cluster_centers
        params._sofa_scores = sofa_scores
        params._metadata = metadata
        return params

    @property
    def tx_mat(self) -> np.ndarray:
        """Transition matrix."""
        return self._tx_mat

    @property
    def r_mat(self) -> np.ndarray:
        """Reward matrix."""
        return self._r_mat

    @property
    def d_0(self) -> np.ndarray:
        """Initial state distribution."""
        return self._d_0

    @property
    def expert_policy(self) -> np.ndarray:
        """Expert policy."""
        return self._expert_policy

    @property
    def admissible_actions(self) -> list[list[int]]:
        """List of admissible actions for each state."""
        return self._admissible_actions

    @property
    def state_cluster_centers(self) -> np.ndarray:
        """Centroids of the state clusters."""
        return self._state_cluster_centers

    @property
    def metadata(self) -> dict:
        """Metadata dictionary."""
        return self._metadata

    @property
    def sofa_scores(self) -> np.ndarray:
        """SOFA scores for each state."""
        return self._sofa_scores

    def save(self, out_dir: Path, /, *,
             save_csv: bool = True,
             save_npz: bool = False):
        """Saves the parameters to the given directory.

        Args:
            out_dir (Path):
                Directory to save the parameters.

            save_csv (bool, optional):
                Whether to save the parameters as CSV files. Defaults to True.

            save_npz (bool, optional):
                Whether to save the parameters as a compressed numpy file.
                Defaults to False.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # delete all files in the directory
        for f in out_dir.iterdir():
            f.unlink()

        if save_npz:
            np.savez_compressed(
                out_dir.joinpath('dynamics.npz'),
                tx_mat=self._tx_mat,
                r_mat=self._r_mat,
                d_0=self._d_0,
                expert_policy=self._expert_policy,
                state_cluster_centers=self._state_cluster_centers,
                sofa_scores=self._sofa_scores)
        if save_csv:
            self.save_mat_as_csv(self._tx_mat, out_dir.joinpath('tx_mat.csv'))
            self.save_mat_as_csv(self._r_mat, out_dir.joinpath('r_mat.csv'))
            self.save_mat_as_csv(self._d_0, out_dir.joinpath('d_0.csv'))
            self.save_mat_as_csv(self._expert_policy,
                                 out_dir.joinpath('expert_policy.csv'))
            self.save_mat_as_csv(self._state_cluster_centers,
                                 out_dir.joinpath('state_cluster_centers.csv'))
            self.save_mat_as_csv(self._sofa_scores,
                                 out_dir.joinpath('sofa_scores.csv'))

        self.write_admissible_actions(
            self._admissible_actions,
            out_dir.joinpath('admissible_actions.txt'))

        with open(out_dir.joinpath('metadata.json'), 'w') as f:
            json.dump(self._metadata, f, indent=4)

    def load_to(self, obj):
        """Loads the parameters to the given object."""
        obj._tx_mat = self.tx_mat
        obj._r_mat = self.r_mat
        obj._d_0 = self.d_0
        obj._expert_policy = self.expert_policy
        obj._admissible_actions = self.admissible_actions
        obj._metadata = self.metadata
        obj._state_cluster_centers = self.state_cluster_centers
        obj._sofa_scores = self.sofa_scores
