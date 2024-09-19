import logging

import numpy as np


def create_valid_dynamics(
        tx_mat: np.ndarray,
        r_mat: np.ndarray,
        d_0: np.ndarray,
        expert_policy: np.ndarray,
        cluster_centers: np.ndarray,
        sofa_scores: np.ndarray, /) -> tuple[np.ndarray, ...]:

    N_s, N_a, N_s_ = tx_mat.shape[0], tx_mat.shape[1], tx_mat.shape[2]
    assert N_s == N_s_, \
        f'Invalid transition matrix shape: N_s ({N_s}) != N_s_ ({N_s_})'
    assert r_mat.shape == (N_s, N_a, N_s), \
        (f'Invalid reward matrix shape: {r_mat.shape}. '
         'Expected {(N_s, N_a, N_s)}')
    assert d_0.shape == (N_s,), \
        f'Invalid initial state distribution shape: {d_0.shape}'
    assert expert_policy.shape == (N_s, N_a), \
        f'Invalid data policy shape: {expert_policy.shape}'
    assert cluster_centers.shape[0] == N_s-2, \
        f'Invalid cluster centers shape: {cluster_centers.shape}'
    assert sofa_scores.shape[0] == N_s-2, \
        f'Invalid sofa scores shape: {sofa_scores.shape}'

    # Add death, survival and s_inf to `cluster_centers`
    cluster_centers_new = np.zeros((N_s+1, cluster_centers.shape[1]))
    cluster_centers_new[:N_s-2, :] = cluster_centers
    logging.debug('Added extra states to `cluster_centers`. '
                  'New shape: %s', cluster_centers_new.shape)

    # Add death, survival and s_inf to `sofa_scores`
    sofa_scores_new = np.zeros(N_s+1)
    sofa_scores_new[:N_s-2] = sofa_scores.squeeze()
    logging.debug('Added extra states to `sofa_scores`. '
                  'New shape: %s', sofa_scores_new.shape)

    # Add the terminal absorbing state `s_inf` to the matrices
    tx_mat_new = np.zeros((N_s+1, N_a, N_s+1))
    tx_mat_new[:N_s, :, :N_s] = tx_mat
    tx_mat = tx_mat_new

    # from death_state, survival_state, and s_inf, always go to s_inf
    tx_mat[N_s-2:, :, N_s] = 1.
    logging.debug('Added `s_inf` to `tx_mat`. New shape: %s', tx_mat.shape)

    r_mat_new = np.zeros_like(tx_mat)
    r_mat_new[:N_s, :, :N_s] = r_mat
    r_mat = r_mat_new
    logging.debug('Added `s_inf` to `r_mat`. New shape: %s', r_mat.shape)

    d_0_new = np.zeros(N_s+1)
    d_0_new[:N_s] = d_0
    d_0 = d_0_new
    logging.debug('Added `s_inf` to `d_0`. New shape: %s', d_0.shape)

    expert_policy_new = np.zeros((N_s+1, N_a))
    expert_policy_new[:N_s, :] = expert_policy
    expert_policy = expert_policy_new
    logging.debug('Added `s_inf` to `expert_policy`. '
                  'New shape: %s', expert_policy.shape)

    # compute list of dead states
    dead_condition = np.isclose(tx_mat.sum(axis=(1,2)), 0.)
    dead_states_list = np.where(dead_condition)[0]

    # while any dead states have nonzero incoming transitions
    i = 1
    while tx_mat[:, :, dead_states_list].max() > 0.:
        logging.debug('Dead state correction round %d: '
                      'number of dead states = %d', i, len(dead_states_list))

        # set probability of transitioning to a dead state to 0
        tx_mat[:, :, dead_states_list] = 0.

        # renormalize transition probabilities
        np.divide(tx_mat, tx_mat.sum(axis=2)[:,:,np.newaxis], 
            out = tx_mat, 
            where = ~np.isclose(tx_mat.sum(axis=2)[:,:,np.newaxis],0.))

        # update dead states list
        dead_condition = np.isclose(tx_mat.sum(axis=(1,2)), 0.)
        dead_states_list = np.where(dead_condition)[0]
        i += 1

    # remove dead states from matrices
    alive_states_list = np.where(~dead_condition)[0]
    tx_mat = tx_mat[alive_states_list, :, :][:, :, alive_states_list]
    logging.debug('Removed dead states from `tx_mat`. New shape: %s',
                  tx_mat.shape)

    r_mat = r_mat[alive_states_list, :, :][:, :, alive_states_list]
    logging.debug('Removed dead states from `r_mat`. New shape: %s',
                  r_mat.shape)

    d_0 = d_0[alive_states_list]

    # renormalize initial state distribution
    d_0 = np.divide(d_0, d_0.sum())
    logging.debug('Removed dead states from `d_0`. New shape: %s',
                  d_0.shape)

    expert_policy = expert_policy[alive_states_list, :]
    logging.debug('Removed dead states from `expert_policy`. New shape: %s',
                  expert_policy.shape)

    cluster_centers = cluster_centers_new[alive_states_list, :]
    logging.debug('Removed dead states from `cluster_centers`. New shape: %s',
                  cluster_centers.shape)

    sofa_scores = sofa_scores_new[alive_states_list]
    logging.debug('Removed dead states from `sofa_scores`. New shape: %s',
                  sofa_scores.shape)

    logging.debug('Dead states removed: %s', dead_states_list)

    # check that the matrices are valid 
    # (TODO: stochastic tx_mat check, expert_policy check)
    assert np.isclose(d_0.sum(), 1.), \
        'Initial state distribution is not stochastic.'
    assert tx_mat.shape[0] == tx_mat.shape[2], \
        'Transition matrix is not square.'
    assert tx_mat.shape == r_mat.shape, \
        'Transition matrix and reward matrix have different shapes.'
    assert tx_mat.shape[0] == d_0.shape[0], \
        ('Transition matrix and initial state distribution have '
         'different shapes.')
    assert tx_mat.shape[:2] == expert_policy.shape, \
        'Transition matrix and data policy have different shapes.'
    assert tx_mat.shape[0] == cluster_centers.shape[0], \
        'Transition matrix and cluster centers have different shapes.'
    assert tx_mat.shape[0] == sofa_scores.shape[0], \
        'Transition matrix and sofa scores have different shapes.'

    return tx_mat, r_mat, d_0, expert_policy, cluster_centers, sofa_scores


def get_num_steps(t:tuple[int, int]) -> tuple[int, int]:
    if t[1] == 0:
        return (t[0], 1)
    return (t[0] + 1, 0)


def create_mapping_order(n, x, y):
    dir_map = {
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
        (1, 0): (0, 1),
        (0, 1): (-1, 0)}

    num_points = n*n

    all_points = set()
    for i in range(n):
        for j in range(n):
            all_points.add((i, j))

    mapping_order = []
    direction = (-1, 0)
    num_steps = (1, 0)
    steps_left = num_steps[0]

    mapping_order.append((x, y))
    num_points -= 1

    while num_points > 0:
        x, y = (x + direction[0], y + direction[1])
        steps_left -= 1
        if (x, y) in all_points:
            mapping_order.append((x, y))
            num_points -= 1

        if steps_left == 0:
            direction = dir_map[direction]
            num_steps = get_num_steps(num_steps)
            steps_left = num_steps[0]

    return mapping_order


def a_num_to_a(A: int, a_num: int) -> tuple[int, int]:
    A_sqrt = round(np.sqrt(A))

    # verify that A is a perfect square
    assert A == A_sqrt**2

    return (a_num // A_sqrt, a_num % A_sqrt)


def a_to_a_num(A: int, a: tuple[int, int]) -> int:
    A_sqrt = round(np.sqrt(A))

    # verify that A is a perfect square
    assert A == A_sqrt**2

    return a[0]*A_sqrt + a[1]


def create_action_map(tx_mat):
    sa_probs = tx_mat.sum(axis=2)
    action_map = np.zeros_like(sa_probs, dtype=int)
    S, A = sa_probs.shape

    A_sqrt = round(np.sqrt(A))
    a_map_order_list = [
        create_mapping_order(A_sqrt, *a_num_to_a(A, a_num))
        for a_num in range(A)]

    for s in range(S):
        for a in range(A):
            a_map_order = a_map_order_list[a]
            for a_map in a_map_order:
                a_map_num = a_to_a_num(A, a_map)
                if np.isclose(sa_probs[s, a_map_num], 1):
                    action_map[s, a] = a_map_num
                    break

    return action_map


def map_inadmissible_actions(
        tx_mat_sparse: np.ndarray, /, *,
        method: str = 'uniform_unweighted', **kwargs) -> np.ndarray:
    if method == 'single':
        a_idx = np.argmax(tx_mat_sparse.sum(axis=2), axis=1)
        s_a_mask = np.isclose(
            tx_mat_sparse.sum(axis=2), 0.).astype(int)[..., np.newaxis]
        a_probs = tx_mat_sparse[
            np.arange(tx_mat_sparse.shape[0]), a_idx, :][:, np.newaxis, :]
        return a_probs * s_a_mask + tx_mat_sparse
    elif method == 'closest':
        tx_mat_full = np.zeros_like(tx_mat_sparse)
        a_map = create_action_map(tx_mat_sparse)
        for s in range(tx_mat_sparse.shape[0]):
            for a in range(tx_mat_sparse.shape[1]):
                tx_mat_full[s, a, :] = tx_mat_sparse[s, a_map[s, a], :]
        return tx_mat_full
    elif method == 'uniform_unweighted':
        s_s_sum = tx_mat_sparse.sum(axis=1)
        n_acts_admissible = tx_mat_sparse.sum(axis=(1,2))
        probs_unwtd = (s_s_sum / n_acts_admissible[:, np.newaxis]
                       )[:, np.newaxis, :]
        s_a_mask = np.isclose(tx_mat_sparse.sum(axis=2), 0.
                              ).astype(int)[..., np.newaxis]
        return probs_unwtd * s_a_mask + tx_mat_sparse
    elif method == 'uniform_weighted':
        raise NotImplementedError(
            '`uniform_weighted` method not implemented yet.')
    else:
        raise ValueError(f'Invalid method: `{method}`')


def get_admissible_actions(tx_mat_sparse: np.ndarray) -> list[list[int]]:
    sa_sum = tx_mat_sparse.sum(axis=(2))
    x, y = np.where(np.isclose(sa_sum, 1.))
    admissible_actions = [[] for _ in range(tx_mat_sparse.shape[0])]
    for s, a in zip(x, y):
        admissible_actions[s].append(a)
    return admissible_actions
