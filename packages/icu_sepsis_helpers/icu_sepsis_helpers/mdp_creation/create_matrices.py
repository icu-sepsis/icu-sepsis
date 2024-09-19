import numpy as np
import pandas as pd
from tqdm import trange


def rl_table_to_unnormalized_matrices(
        rl_table: pd.DataFrame,
        n_states: int,
        n_action_levels: int, *,
        r_death: float = -1.0,
        r_survive: float = 1.0,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # tx_mat
    tx_mat = np.zeros((n_states+2, n_action_levels**2, n_states+2))
    d_0 = np.zeros((n_states+2,))
    expert_policy = np.zeros((n_states+2, n_action_levels**2))
    r_mat = np.zeros_like(tx_mat)

    row = rl_table.iloc[0, :]
    for i in trange(1, len(rl_table)):
        row_next = rl_table.iloc[i, :]
        b, s, a = row['bloc'], row['state'], row['action']
        s_, b_ = row_next['state'], row_next['bloc']
        expert_policy[s, a] += 1

        # start of episode
        if b == 1:
            d_0[s] += 1

        # one step in the episode
        if b_ == b+1:
            assert row_next['bloc'] > row['bloc'], \
                (f'blocs {row["bloc"]}, '
                 f'{row_next["bloc"]} not aligned for '
                 f'icustayid {row["icustayid"]}')
            tx_mat[s, a, s_] += 1

        # end of episode. will transition to dead (idx n_states) or
        # alive state (idx n_states+1)
        else:
            next_state = n_states if row['outcome_y'] == 1 else n_states+1
            tx_mat[s, a, next_state] += 1

        row = row_next

    r_mat[:, :, n_states] = r_death
    r_mat[:, :, n_states+1] = r_survive

    return tx_mat, r_mat, d_0, expert_policy


def normalize_tx_mat(
        tx_mat: np.ndarray,
        transition_threshold: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # normalize tx_mat
    tx_mat_copy = tx_mat.copy()
    s_a_mat = tx_mat_copy.sum(axis=2)
    below_thresh = np.where(s_a_mat <= transition_threshold)
    tx_mat_copy[below_thresh] = 0
    tx_mat_copy = np.divide(
        tx_mat_copy,
        tx_mat_copy.sum(axis=2, keepdims=True),
        out=np.zeros_like(tx_mat_copy),
        where=tx_mat_copy.sum(axis=2, keepdims=True) > 0)

    return tx_mat_copy


def normalize_d_0(
        d_0: np.ndarray,
        ) -> np.ndarray:
    return d_0 / d_0.sum()


def normalize_expert_policy(
        expert_policy: np.ndarray,
        ) -> np.ndarray:
    return np.divide(
        expert_policy,
        np.sum(expert_policy, axis=1, keepdims=True),
        out=np.zeros_like(expert_policy),
        where=np.sum(expert_policy, axis=1, keepdims=True) > 0)
