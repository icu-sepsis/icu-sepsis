import logging
from pathlib import Path
import tempfile

import pandas as pd
from icu_sepsis_helpers.mdp_creation.create_matrices import (
    normalize_d_0, normalize_expert_policy, normalize_tx_mat,
    rl_table_to_unnormalized_matrices)
from icu_sepsis_helpers.mdp_creation.create_rl_table import create_rl_dataset
from icu_sepsis_helpers.mdp_creation.parse_matrices import (
    create_valid_dynamics, get_admissible_actions, map_inadmissible_actions)
from icu_sepsis.utils.io import MDPParameters


def build_mimic_params(
        dataset_path: str | Path, out_dir: str | Path, n_states: int,
        n_action_levels: int, threshold: int, /, *,
        seed: int = 0, r_survive: float = 1.0, r_death: float = 0.0,
        action_map_method: str = 'uniform_unweighted', temp_dir: Path = None,
        save_npz: bool = True, save_csv: bool = True, n_clustering: int = 32,
        ratio_clustering: float = 0.25, max_iter_kmeans: int = 10_000,
        init_kmeans: str = 'k-means++'):

    metadata = {
        'n_states': n_states,
        'n_actions': n_action_levels**2,
        'r_survive': r_survive,
        'r_death': r_death,
        'threshold': threshold,
        'seed': seed,
        'action_map_method': action_map_method
    }

    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)

    with tempfile.mkdtemp(prefix='mimic_') as td:
        temp_dir = Path(td)

        create_rl_dataset(
            dataset_path, temp_dir, n_states, n_action_levels,
            seed=seed, ratio_clustering=ratio_clustering,
            max_iter=max_iter_kmeans, init=init_kmeans,
            n_clustering=n_clustering)
        logging.info('Created RL table in %s', temp_dir)

        # Load the RL table
        mimic_rl_table_path = temp_dir.joinpath('mimic_rl_table.csv')
        mimic_rl_table = pd.read_csv(mimic_rl_table_path)
        logging.info('Loaded RL table form %s', mimic_rl_table_path)

        # Create the unnormalized matrices
        (
            tx_mat_u, r_mat, d_0_u, expert_policy_u
        ) = rl_table_to_unnormalized_matrices(mimic_rl_table, n_states, 
                                              n_action_levels,
                                              r_survive=r_survive,
                                              r_death=r_death)
        logging.info('Created unnormalized matrices')

        # Normalize the matrices
        d_0 = normalize_d_0(d_0_u)
        expert_policy = normalize_expert_policy(expert_policy_u)
        tx_mat_sparse = normalize_tx_mat(tx_mat_u, threshold)
        logging.info('Normalized matrices')

        # Create valid dynamics
        cluster_centers_u = pd.read_csv(
            temp_dir.joinpath('mimic_cluster_centers.csv')
        ).values
        sofa_scores_u = pd.read_csv(
            temp_dir.joinpath('mimic_sofa.csv'))

    (
        tx_mat_sparse, r_mat, d_0, expert_policy,
        cluster_centers, sofa_scores
    ) = create_valid_dynamics(
        tx_mat_sparse, r_mat, d_0, expert_policy,
        cluster_centers_u, sofa_scores_u)
    logging.info('Created valid dynamics')

    # Create the list of admissible actions
    admissible_actions = get_admissible_actions(tx_mat_sparse)
    logging.info('Created list of admissible actions')

    # Map inadmissible actions
    tx_mat = map_inadmissible_actions(tx_mat_sparse, method=action_map_method)
    logging.info('Mapped inadmissible actions')

    # Save the dynamics
    params = MDPParameters.create(
        tx_mat, r_mat, d_0,
        expert_policy, admissible_actions,
        cluster_centers, sofa_scores, metadata)

    params.save(out_dir, save_npz=save_npz, save_csv=save_csv)
    logging.info('Saved parameters to %s', out_dir)
