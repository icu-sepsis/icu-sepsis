import itertools
import numpy as np

from icu_sepsis_helpers.mdp_creation.parse_matrices import \
    map_disallowed_actions


def perturb(tx_mat, allowed_actions, sigma):
    nS = tx_mat.shape[0]
    ac_tuples = []
    for s in range(nS-3):
        ac_tuples.append([(s, a) for a in allowed_actions[s]])
    ac_combinations = list(itertools.chain.from_iterable(ac_tuples))
    n = len(ac_combinations)

    # Choose to drop each action with probability sigma
    drop = np.random.rand(n) < sigma
    drop = drop.astype(int)

    remaining_actions = np.array(ac_combinations)[np.where(drop == 0)]
    remaining_actions = [(s, a) for (s, a) in remaining_actions]
    allowed_actions_new = []
    for s in range(nS-3):
        allowed_actions_new.append([a for a in allowed_actions[s]
                                    if (s, a) in remaining_actions])

    # For states with no allowed actions, randomly choose one that is allowed
    for s in range(nS-3):
        if len(allowed_actions_new[s]) == 0:
            allowed_actions_new[s] = [np.random.choice(allowed_actions[s])]

    # Keep all actions in the last 3 states
    for s in range(nS-3, nS):
        allowed_actions_new.append(allowed_actions[s])

    n_new = sum([len(allowed_actions_new[s]) for s in range(nS)])

    for i in range(nS-3, nS):
        n += len(allowed_actions[i])
    print(f'Dropped {100*(n-n_new)/n:.2f}% of actions')

    # Create new transition matrix
    tx_mat_new = tx_mat.copy()
    for s in range(nS-3):
        for a in range(tx_mat.shape[1]):
            if a not in allowed_actions_new[s]:
                tx_mat_new[s, a, :] = 0

    # Renormalize
    tx_mat_new = map_disallowed_actions(tx_mat_new,
                                        method='uniform_unweighted')

    return tx_mat_new, allowed_actions_new
