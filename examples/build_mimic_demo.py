import logging
import argparse

# Choose if you want to use gym or gymnasium
import gymnasium as gym
# import gym

# Import needed to put the environment in the gym registry
import icu_sepsis

# Helper functions
from icu_sepsis_helpers.build import build_mimic_params
from icu_sepsis_helpers.baselines import get_mdp_stats
from icu_sepsis.utils.io import MDPParameters

ENV_ID = 'Sepsis/ICU-Sepsis-v1'

# Give arg values in command line or set them here
ARGS = {
    'mimic_dataset_in_path'     : 'data/mimic_dataset_table.csv',
    'output_dir'                : 'data/output/mimic_params_test/',
    'n_states'                  : 750,
    'n_action_levels'           : 5,        # n_actions = n_action_levels ** 2
    'threshold'                 : 20,       # Threshold for min no. of samples in (s,a) pair
    'seed'                      : 0
}

def main(args:argparse.Namespace):

    # build the MDP parameters
    logging.info('Building MDP parameters')
    build_mimic_params(args.input_path, args.output_dir,
        args.n_states, args.n_action_levels,
        args.threshold)

    # load the MDP parameters
    params = MDPParameters(args.output_dir)

    # create the environment
    env = gym.make(ENV_ID, params=params)

    # print the metadata
    print(env.unwrapped.get_metadata())

    # demo usage
    state, info = env.reset()
    print(state, info)

    state, rew, terminated, truncated, info = env.step(0)
    print(state, rew, terminated, truncated, info)

    # get some baseline MDP stats
    stats = get_mdp_stats(env)
    print(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', 
        default=ARGS['mimic_dataset_in_path'], 
        help='Path to the input dataset')
    parser.add_argument('-o', '--output-dir', 
        default=ARGS['output_dir'], 
        help='Path to the output directory')
    parser.add_argument('-s', '--n-states', 
        type=int, 
        default=ARGS['n_states'], 
        help='Number of states')
    parser.add_argument('-a', '--n-action-levels', 
        type=int, 
        default=ARGS['n_action_levels'], 
        help='Number of action levels')
    parser.add_argument('-t', '--threshold', 
        type=int, 
        default=ARGS['threshold'], 
        help='Threshold for min no. of samples in (s,a) pair')
    parser.add_argument('--seed',
        type=int, 
        default=ARGS['seed'],
        help='Random seed to create the MDP parameters')

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(args)
