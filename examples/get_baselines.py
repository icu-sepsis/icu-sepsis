"""Compute baseline statistics for the ICU Sepsis environment."""

import argparse
import gymnasium as gym
import icu_sepsis
from icu_sepsis_helpers import get_mdp_stats


def main(env_ver: int, **kwargs):
    env = gym.make(f'Sepsis/ICU-Sepsis-v{env_ver}', **kwargs)

    stats = get_mdp_stats(env)
    for policy, stats in stats.items():
        print(f'{policy}:')
        for key, val in stats.items():
            print(f'  {key}: {val}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=2,
                        help='Environment version')
    parser.add_argument('-i', type=str, default=None,
                        help='Inadmissible action strategy')
    args = parser.parse_args()

    main_kwargs = {'env_ver': args.v}
    if args.i is not None:
        main_kwargs['inadmissible_action_strategy'] = args.i

    main(**main_kwargs)
