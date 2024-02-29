import gymnasium
import numpy as np
from icu_sepsis_helpers.utils.mdp import estimate_j_pi, value_iteration


def get_mdp_stats(env: gymnasium.Env, gamma: float = 1.0) -> dict:
    stats = {}
    dynamics = env.unwrapped.get_dynamics()
    rand_policy = lambda x: np.random.randint(
        0, env.unwrapped.get_metadata()['n_actions'])
    data_policy_arr = env.unwrapped.data_policy
    data_policy = lambda s: np.random.choice(
        data_policy_arr.shape[1], p=data_policy_arr[s, :])
    pi_star_arr, _ = value_iteration(
        dynamics['tx_mat'], dynamics['r_mat'], gamma)
    pi_star = lambda x: pi_star_arr[x]

    stats['pi_star'] = {}
    j_star, j_dict_star = estimate_j_pi(env, pi_star, desc='J_optm')
    stats['pi_star']['j_mean'] = j_star
    stats['pi_star']['ep_len'] = j_dict_star['mean_ep_len']

    stats['rand_policy'] = {}
    j_rand, j_dict_rand = estimate_j_pi(env, rand_policy, desc='J_rand')
    stats['rand_policy']['j_mean'] = j_rand
    stats['rand_policy']['ep_len'] = j_dict_rand['mean_ep_len']

    stats['data_policy'] = {}
    j_data, j_dict_data = estimate_j_pi(env, data_policy, desc='J_data')
    stats['data_policy']['j_mean'] = j_data
    stats['data_policy']['ep_len'] = j_dict_data['mean_ep_len']

    print()
    return stats
