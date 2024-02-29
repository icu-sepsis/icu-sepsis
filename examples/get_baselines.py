import gymnasium as gym
import icu_sepsis
from icu_sepsis_helpers import get_mdp_stats


def main():
    env = gym.make('Sepsis/ICU-Sepsis-v1')

    stats = get_mdp_stats(env)
    print(stats)


if __name__ == '__main__':
    main()
