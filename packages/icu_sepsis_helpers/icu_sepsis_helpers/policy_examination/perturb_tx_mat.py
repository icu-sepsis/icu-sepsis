import itertools
import numpy as np

from icu_sepsis_helpers.mdp_creation.parse_matrices import \
    map_inadmissible_actions


def perturb(tx_mat, admissible_actions, sigma):
    nS = tx_mat.shape[0]
    ac_tuples = []
    for s in range(nS-3):
        ac_tuples.append([(s, a) for a in admissible_actions[s]])
    ac_combinations = list(itertools.chain.from_iterable(ac_tuples))
    n = len(ac_combinations)

    # Choose to drop each action with probability sigma
    drop = np.random.rand(n) < sigma
    drop = drop.astype(int)

    remaining_actions = np.array(ac_combinations)[np.where(drop == 0)]
    remaining_actions = [(s, a) for (s, a) in remaining_actions]
    admissible_actions_new = []
    for s in range(nS-3):
        admissible_actions_new.append([a for a in admissible_actions[s]
                                    if (s, a) in remaining_actions])

    # For states with no admissible actions, randomly choose one that is admissible
    for s in range(nS-3):
        if len(admissible_actions_new[s]) == 0:
            admissible_actions_new[s] = [np.random.choice(admissible_actions[s])]

    # Keep all actions in the last 3 states
    for s in range(nS-3, nS):
        admissible_actions_new.append(admissible_actions[s])

    n_new = sum([len(admissible_actions_new[s]) for s in range(nS)])

    for i in range(nS-3, nS):
        n += len(admissible_actions[i])
    print(f'Dropped {100*(n-n_new)/n:.2f}% of actions')

    # Create new transition matrix
    tx_mat_new = tx_mat.copy()
    for s in range(nS-3):
        for a in range(tx_mat.shape[1]):
            if a not in admissible_actions_new[s]:
                tx_mat_new[s, a, :] = 0

    # Renormalize
    tx_mat_new = map_inadmissible_actions(tx_mat_new,
                                        method='uniform_unweighted')

    return tx_mat_new, admissible_actions_new
