import numpy as np
from tqdm import trange
import pickle
from matplotlib import pyplot as plt


def check_convergence(vals: list, steps: list, delta: float = 0.01):
    """
    Return the number of trials required to reach convergence
    """

    # find the point where the mean of last 1k trials is within delta
    # of the mean of last 10k trials
    num_steps = -1
    last_1k = -1
    last_10k = -1
    for i in trange(10000, len(vals), 100):
        last_1k = np.mean(vals[i-1000:i])
        last_10k = np.mean(vals[-10000:])
        if abs(last_1k - last_10k)/last_10k < delta:
            num_steps = i
            break

    total_steps = sum(steps[:num_steps])

    return num_steps, total_steps, last_1k


def load_vals(filename: str):
    """
    Load values from a file where each line is a value
    """
    with open(filename, 'r') as f:
        vals = f.readlines()
        vals = [float(val) for val in vals]
    return vals


def load_pcsd_dict(filename: str):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    print(data.keys())
    return data


MAPPING = {
    'ppo': 'PPO',
    'sac': 'SAC',
    'qlearning': 'Q-Learning',
    'dqn': 'DQN',
    'sarsa': 'Sarsa'
}


def plot_returns(algos: list):
    """
    Plot the returns of different algorithms
    """
    fig, ax = plt.subplots()
    S = 500
    for algo in algos:
        means = load_vals(f'data/{algo}_returns_mean.txt')
        means_smooth = np.convolve(means, np.ones(1000)/1000, mode='same')[S+1:-(S+1)]
        stds = load_vals(f'data/{algo}_returns_stderr.txt')[S+1:-(S+1)]
        x = np.arange(0, len(means_smooth)/1000, 1/1000)
        ax.fill_between(x, np.array(means_smooth) - np.array(stds),
                        np.array(means_smooth) + np.array(stds), alpha=0.4)
        ax.plot(x, means_smooth, label=MAPPING[algo])
        ax.set_title('Average Return')
        ax.set_ylim(0.75, 0.9)
        ax.set_xlabel('Episodes (K)')
        ax.set_ylabel('Average Return')
    ax.axhline(0.78, color='black', linestyle='--', linewidth=0.5, label='Expert')
    ax.axhline(0.88, color='black', linestyle='-.', linewidth=0.5, label='Optimal') 
    ax.legend()
    fig.savefig('returns.png')


def plot_steps(algos: list):
    """
    Plot the steps of different algorithms
    """
    fig, ax = plt.subplots()
    S = 500
    for algo in algos:
        means = load_vals(f'data/{algo}_num_steps_mean.txt')
        means_smooth = np.convolve(means, np.ones(1000)/1000, mode='same')[S+1:-(S+1)]
        stds = load_vals(f'data/{algo}_num_steps_stderr.txt')[S+1:-(S+1)]
        x = np.arange(0, len(means_smooth)/1000, 1/1000)
        ax.fill_between(x, np.array(means_smooth) - np.array(stds),
                        np.array(means_smooth) + np.array(stds), alpha=0.4)
        ax.plot(x, means_smooth, label=MAPPING[algo])
        ax.set_title('Average Steps')
        ax.set_xlabel('Episodes (K)')
        ax.set_ylabel('Average Steps')
    ax.axhline(13.27, color='black', linestyle='--', linewidth=0.5, label='Dataset')
    ax.legend()
    fig.savefig('steps.png')


def main():
    algos = ['ppo', 'sac', 'qlearning', 'dqn', 'sarsa']
    # for algo in algos:
    #     vals = load_vals(f'data/{algo}_returns_mean.txt')
    #     steps = load_vals(f'data/{algo}_num_steps_mean.txt')
    #     trials, tot, j = check_convergence(vals, steps, 1e-4)
    #     print(f"{algo} Convergence after {trials/1000:.2f}K trials")
    #     print(f"\tTotal steps: {tot/1e6:.2f}M")
    #     print(f"\tLast 1k mean: {j:.2f}")
    plot_returns(algos)
    plot_steps(algos)


if __name__ == "__main__":
    main()
