from pathlib import Path
import gymnasium as gym
import icu_sepsis
from icu_sepsis.utils.io import MDPParameters
from icu_sepsis_helpers import get_mdp_stats
from icu_sepsis_helpers.policy_examination import perturb
from multiprocessing import Pool
import json
import numpy as np
from matplotlib import pyplot as plt


def run_trial(params_path, sigma):
    params = MDPParameters(params_path)
    tx_mat = params.tx_mat
    admissible_actions = params.admissible_actions

    tx_mat_perturbed, admissible_actions_perturbed = perturb(
        tx_mat, admissible_actions, sigma)

    params._tx_mat = tx_mat_perturbed
    params._admissible_actions = admissible_actions_perturbed

    env_perturbed = gym.make('Sepsis/ICU-Sepsis-v1', params=params)

    stats_perturbed = get_mdp_stats(env_perturbed)

    return stats_perturbed


def get_results(param_dirname, sigma_vals, n_trials):
    params_path = Path(f'mimic_params/{param_dirname}')

    for sigma in sigma_vals:
        print(f'Perturbing with sigma={sigma}')

        results = {
            'pi_star': {
                'j_mean': [], 'ep_len': []},
            'rand_policy': {
                'j_mean': [], 'ep_len': []},
            'expert_policy': {
                'j_mean': [], 'ep_len': []}}

        with Pool(16) as pool:
            result_list = pool.starmap(
                run_trial, [(params_path, sigma)]*n_trials)

            for result in result_list:
                results['pi_star']['j_mean'].append(
                    result['pi_star']['j_mean'])
                results['pi_star']['ep_len'].append(
                    result['pi_star']['ep_len'])
                results['rand_policy']['j_mean'].append(
                    result['rand_policy']['j_mean'])
                results['rand_policy']['ep_len'].append(
                    result['rand_policy']['ep_len'])
                results['expert_policy']['j_mean'].append(
                    result['expert_policy']['j_mean'])
                results['expert_policy']['ep_len'].append(
                    result['expert_policy']['ep_len'])

        with open(
            f'perturbed_results_{param_dirname}_sigma_{sigma}.json', 'w'
        ) as f:
            json.dump(results, f, indent=4)


def aggregate_results(param_dirname, sigma_vals):
    results = {
        'pi_star': {
            'rewards': {'mean': [], 'std': []},
            'ep_len': {'mean': [], 'std': []}},
        'rand_policy': {
            'rewards': {'mean': [], 'std': []},
            'ep_len': {'mean': [], 'std': []}},
        'expert_policy': {
            'rewards': {'mean': [], 'std': []},
            'ep_len': {'mean': [], 'std': []}}}

    for sigma in sigma_vals:
        with open(
            f'perturbed_results_{param_dirname}_sigma_{sigma}.json', 'r'
        ) as f:
            res = json.load(f)

        for policy in results.keys():
            results[policy]['rewards']['mean'].append(
                np.mean(res[policy]['j_mean']))
            results[policy]['rewards']['std'].append(
                np.std(res[policy]['j_mean']))
            results[policy]['ep_len']['mean'].append(
                np.mean(res[policy]['ep_len']))
            results[policy]['ep_len']['std'].append(
                np.std(res[policy]['ep_len']))

    return results


def plot_results_per_pol(param_dirname_list, sigma_vals):
    all_results = {}
    for param_dirname in param_dirname_list:
        all_results[param_dirname] = aggregate_results(param_dirname,
                                                       sigma_vals)

    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))

    for i, policy in enumerate(all_results[param_dirname_list[0]].keys()):
        for param_dirname in param_dirname_list:
            results = all_results[param_dirname]
            axs1[i].errorbar(sigma_vals, results[policy]['rewards']['mean'],
                             yerr=results[policy]['rewards']['std'],
                             label=param_dirname)

            axs2[i].errorbar(sigma_vals, results[policy]['ep_len']['mean'],
                             yerr=results[policy]['ep_len']['std'],
                             label=param_dirname.split('_', 6)[-1])

        axs1[i].set_title(policy)
        axs1[i].set_xlabel('Sigma')
        axs1[i].set_ylabel('Mean reward')
        axs1[i].set_ylim(0.6, 1.0)
        axs1[i].axhline(0.77, color='black', linestyle='--', label='Dataset',
                        linewidth=0.5)
        axs1[i].legend()

        axs2[i].set_title(policy)
        axs2[i].set_xlabel('Sigma')
        axs2[i].set_ylabel('Mean episode length')
        axs2[i].set_ylim(0, 50)
        axs2[i].axhline(13.27, color='black', linestyle='--', label='Dataset',
                        linewidth=0.5)
        axs2[i].legend()

    fig1.savefig('perturbed_results.png')
    fig2.savefig('perturbed_results_ep_len.png')


def plot_results_per_env(param_dirname_list, sigma_vals):
    all_results = {}
    for param_dirname in param_dirname_list:
        all_results[param_dirname] = aggregate_results(param_dirname,
                                                       sigma_vals)

    fig1, axs1 = plt.subplots(1, len(param_dirname_list), figsize=(15, 5))
    fig2, axs2 = plt.subplots(1, len(param_dirname_list), figsize=(15, 5))

    for i, param_dirname in enumerate(param_dirname_list):
        results = all_results[param_dirname]
        for policy in results.keys():
            axs1[i].errorbar(sigma_vals, results[policy]['rewards']['mean'],
                             yerr=results[policy]['rewards']['std'],
                             label=policy)

            axs2[i].errorbar(sigma_vals, results[policy]['ep_len']['mean'],
                             yerr=results[policy]['ep_len']['std'],
                             label=policy)

        axs1[i].set_title(param_dirname.split('_', 5)[-1])
        axs1[i].set_xlabel('Sigma')
        axs1[i].set_ylabel('Mean reward')
        axs1[i].set_ylim(0.6, 1.0)
        axs1[i].axhline(0.77, color='black', linestyle='--', label='Dataset',
                        linewidth=0.5)
        axs1[i].legend()

        axs2[i].set_title(param_dirname.split('_', 5)[-1])
        axs2[i].set_xlabel('Sigma')
        axs2[i].set_ylabel('Mean episode length')
        axs2[i].set_ylim(0, 50)
        axs2[i].axhline(13.27, color='black', linestyle='--', label='Dataset',
                        linewidth=0.5)
        axs2[i].legend()

    fig1.savefig('perturbed_results_per_env.png')
    fig2.savefig('perturbed_results_ep_len_per_env.png')


def main():
    param_dirnames = [
        'states_750_action_lvls_5_thr_5',
        'states_750_action_lvls_5_thr_20']
    sigma_vals = [0.01, 0.05, 0.2, 0.5, 0.8]

    plot_results_per_env(param_dirnames, sigma_vals)


if __name__ == '__main__':
    main()
