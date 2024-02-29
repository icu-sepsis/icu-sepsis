import gymnasium as gym
import icu_sepsis


def main():
    env = gym.make('Sepsis/ICU-Sepsis-v1')

    state, info = env.reset()
    print('Initial state:', state)

    next_state, reward, terminated, truncated, info = env.step(0)
    print('Next state:', next_state)
    print('Reward:', reward)
    print('Terminated:', terminated)
    print('Truncated:', truncated)


if __name__ == '__main__':
    main()
