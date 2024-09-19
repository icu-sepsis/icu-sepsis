"""Baselines for the ICU-Sepsis environment."""

import gymnasium
import numpy as np
from icu_sepsis_helpers.utils.mdp import estimate_j_pi, value_iteration


def get_mdp_stats(env: gymnasium.Env, gamma: float = 1.0) -> dict:
    """Get MDP statistics for the environment."""

    stats = {}
    dynamics = env.unwrapped.dynamics
    expert_policy_arr = env.unwrapped.expert_policy
    pi_star_arr, _ = value_iteration(
        dynamics['tx_mat'], dynamics['r_mat'], gamma)

    def _rand_policy(s: int) -> int:
        return np.random.randint(0, env.unwrapped.env_metadata['n_actions'])

    def _expert_policy(s: int) -> int:
        return np.random.choice(expert_policy_arr.shape[1],
                                p=expert_policy_arr[s, :])

    def _pi_star(s: int) -> int:
        return pi_star_arr[s]

    stats['pi_star'] = {}
    j_star, j_dict_star = estimate_j_pi(env, _pi_star, desc='J_optm')
    stats['pi_star']['j_mean'] = j_star
    stats['pi_star']['ep_len'] = j_dict_star['mean_ep_len']

    stats['rand_policy'] = {}
    j_rand, j_dict_rand = estimate_j_pi(env, _rand_policy, desc='J_rand')
    stats['rand_policy']['j_mean'] = j_rand
    stats['rand_policy']['ep_len'] = j_dict_rand['mean_ep_len']

    stats['expert_policy'] = {}
    j_data, j_dict_data = estimate_j_pi(env, _expert_policy, desc='J_data')
    stats['expert_policy']['j_mean'] = j_data
    stats['expert_policy']['ep_len'] = j_dict_data['mean_ep_len']

    print()
    return stats
