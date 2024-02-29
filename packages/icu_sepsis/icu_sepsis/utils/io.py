import json
from pathlib import Path

import numpy as np
import pandas as pd


class MDPParameters:
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

        self._allowed_actions = self.read_allowed_actions(
            data_dir.joinpath('allowed_actions.txt'))

    @staticmethod
    def read_allowed_actions(infile_path: Path) -> list[list[int]]:
        n_acs = []
        allowed_actions = []
        with open(infile_path, 'r') as f:
            # Read number of allowed actions in the first line
            line_items = f.readline().strip().split()
            n_acs = [int(n) for n in line_items]

            # Read allowed actions for each state in the following lines
            for n in n_acs:
                line_items = f.readline().strip().split()
                assert len(line_items) == n, \
                    (f'Number of allowed actions ({n}) does not match the '
                     f'number of actions in the line ({len(line_items)}).')
                allowed_actions.append([int(ac) for ac in line_items])

        return allowed_actions

    @staticmethod
    def write_allowed_actions(allowed_actions: list[list[int]],
                              outfile_path: Path):
        with open(outfile_path, 'w') as f:
            # write number of allowed actions in the first line
            for acs in allowed_actions:
                f.write(f'{len(acs)} ')
            f.write('\n')

            # write allowed actions for each state in the following lines
            for acs in allowed_actions:
                for ac in acs:
                    f.write(f'{ac} ')
                f.write('\n')

    @staticmethod
    def save_mat_as_csv(mat: np.ndarray, out_path: Path, /):
        mat_flat = mat.reshape(-1, mat.shape[-1])
        pd.DataFrame(mat_flat).to_csv(out_path, index=False, header=False)

    @staticmethod
    def create(tx_mat, r_mat, d_0, expert_policy, allowed_actions,
               state_cluster_centers, sofa_scores, metadata):
        params = MDPParameters.__new__(MDPParameters)
        params._tx_mat = tx_mat
        params._r_mat = r_mat
        params._d_0 = d_0
        params._expert_policy = expert_policy
        params._allowed_actions = allowed_actions
        params._state_cluster_centers = state_cluster_centers
        params._sofa_scores = sofa_scores
        params._metadata = metadata
        return params

    @property
    def tx_mat(self) -> np.ndarray:
        return self._tx_mat

    @property
    def r_mat(self) -> np.ndarray:
        return self._r_mat

    @property
    def d_0(self) -> np.ndarray:
        return self._d_0

    @property
    def expert_policy(self) -> np.ndarray:
        return self._expert_policy

    @property
    def allowed_actions(self) -> list[list[int]]:
        return self._allowed_actions

    @property
    def state_cluster_centers(self) -> np.ndarray:
        return self._state_cluster_centers

    @property
    def metadata(self) -> dict:
        return self._metadata

    def save(self, out_dir: Path, /, *,
             save_csv: bool = True,
             save_npz: bool = False):
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

        self.write_allowed_actions(self._allowed_actions,
                                   out_dir.joinpath('allowed_actions.txt'))

        with open(out_dir.joinpath('metadata.json'), 'w') as f:
            json.dump(self._metadata, f, indent=4)
