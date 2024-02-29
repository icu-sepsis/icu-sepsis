import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, zscore
from sklearn.cluster import KMeans


def create_rl_dataset(MIMIC_dataset_path: Path, output_root:Path,
        n_states: int, n_action_levels: int, /, *,
        n_clustering: int = 32, ratio_clustering: float = 0.25, seed: int = 0,
        max_iter: int = 10_000, init: str = 'k-means++'):
    MIMIC_Dataset_Raw = pd.read_csv(MIMIC_dataset_path)
    logging.debug('Dataset shape: %s', MIMIC_Dataset_Raw.shape)

    # TODO: Deal with null values
    MIMIC_Dataset_Raw.fillna(0, inplace=True)

    # binary
    colbin = ['gender', 'mechvent', 'max_dose_vaso', 're_admission']

    # normal
    colnorm = ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP',
               'RR', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride',
               'Glucose', 'Magnesium', 'Calcium', 'Hb', 'WBC_count',
               'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2',
               'Arterial_BE', 'HCO3', 'Arterial_lactate', 'SOFA', 'SIRS',
               'Shock_Index', 'PaO2_FiO2', 'cumulated_balance']

    # logarithmic
    collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR',
              'input_total', 'input_4hourly', 'output_total', 'output_4hourly']

    all_cols = colbin + colnorm + collog

    MIMIC_Dataset = create_features(MIMIC_Dataset_Raw, colbin, colnorm, collog)

    save_rl_dataset(
        MIMIC_Dataset, MIMIC_Dataset_Raw, n_states, n_action_levels,
        all_cols, n_clustering, ratio_clustering, seed, output_root,
        max_iter=max_iter, init=init)


def normalize_dataset(
        dataset_raw: pd.DataFrame, colbin: list[str],
        colnorm: list[str], collog: list[str], /) -> pd.DataFrame:
    dataset_bin = dataset_raw[colbin]
    dataset_norm = dataset_raw[colnorm]
    dataset_log = dataset_raw[collog]

    # normalizing based on type of data
    dataset = dataset_raw.copy()
    dataset[colbin] = dataset_bin - 0.5
    dataset[colnorm] = zscore(dataset_norm)
    dataset[collog] = zscore(np.log(dataset_log+0.1))

    logging.info('Normalized dataset: %s', dataset.shape)

    return dataset


def create_features(
        Dataset_Raw: pd.DataFrame, colbin, colnorm, collog, /) -> pd.DataFrame:
    all_cols = colbin + colnorm + collog
    Dataset = normalize_dataset(Dataset_Raw, colbin, colnorm, collog)

    # TODO: is this correct?
    Dataset[all_cols[4-1]] = Dataset[all_cols[4-1]] + 0.6  # max dose?
    Dataset[all_cols[45-1]] = Dataset[all_cols[45-1]] * 2  # increase weight

    # column for reward
    Dataset['outcome_y'] = Dataset['mortality_90d'].astype(int)

    return Dataset


def get_action_nums(Dataset: pd.DataFrame, action_name: str, num_actions: int):
    ac = Dataset[action_name]
    l_ac = len(ac[ac > 0])
    ac_norm = rankdata(ac[ac > 0])/l_ac
    ac_vals = (ac_norm * (num_actions-1) + 1 - 1e-8).astype(int)
    ac_s = pd.Series(0, index=ac.index)
    ac_s[ac > 0] = ac_vals
    return ac_s


def cluster_states(
        Dataset: pd.DataFrame, cols: list[str], n_states: int,
        n_clustering: int, ratio_clustering: float, seed: int, /, *,
        max_iter: int = 10_000, init: str = 'k-means++'
        ) -> tuple[np.ndarray, dict]:

    logging.info('Runnig K-means clustering.')
    X_sample_clustering = Dataset[cols].sample(
        frac=ratio_clustering, random_state=seed)
    kmeans_fit = KMeans(
        n_clusters=n_states, max_iter=max_iter,
        init=init, n_init=n_clustering, random_state=seed
    ).fit(X_sample_clustering)

    clusters_df = pd.DataFrame(kmeans_fit.cluster_centers_, columns=cols)
    data = {
        'cluster_centers': clusters_df}

    logging.info('Clustered states: %s', clusters_df.shape)
    return kmeans_fit.predict(Dataset[cols]), data


def save_rl_dataset(
        Dataset: pd.DataFrame, Dataset_Raw: pd.DataFrame, n_states: int,
        n_action_levels: int, cols: list[str], n_clustering: int,
        ratio_clustering: float, seed: int, output_root: Path, /, *,
        max_iter: int = 10_000, init: str = 'k-means++'):
    output_root.mkdir(parents=True, exist_ok=True)

    ac1 = get_action_nums(Dataset_Raw, 'input_4hourly', n_action_levels)
    ac2 = get_action_nums(Dataset_Raw, 'max_dose_vaso', n_action_levels)
    Dataset['action'] = n_action_levels * ac1 + ac2

    Dataset['state'], extra_data = cluster_states(
        Dataset, cols, n_states, n_clustering, ratio_clustering,
        seed, max_iter=max_iter, init=init)

    cluster_path = output_root.joinpath('mimic_cluster_centers.csv')
    extra_data['cluster_centers'].to_csv(cluster_path, index=False)
    logging.debug('Cluster centers saved to %s', cluster_path)

    dataset_path = output_root.joinpath('mimic_rl_table.csv')
    rl_dataset = Dataset[['bloc', 'state', 'action', 'outcome_y']]
    rl_dataset.to_csv(dataset_path, index=False)
    logging.info('Dataset saved to %s', dataset_path)

    sofa_path = output_root.joinpath('mimic_sofa.csv')
    rl_dataset['SOFA'] = Dataset_Raw['SOFA']
    avg_sofa = rl_dataset.groupby('state')['SOFA'].mean()
    avg_sofa.to_csv(sofa_path, index=False)
    logging.debug('SOFA saved to %s', sofa_path)
