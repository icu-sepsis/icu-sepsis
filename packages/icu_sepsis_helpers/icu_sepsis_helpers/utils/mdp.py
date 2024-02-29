from math import factorial
from typing import Callable

import gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange


def value_iteration(tx_mat, r_mat, gamma, 
        /, *, 
        max_steps:int=50_000,
        delta:float=1e-6
    )-> tuple[np.ndarray, np.ndarray]:
    """Perform value iteration to find the optimal value function. Return the optimal policy and the value function."""
    N_s = tx_mat.shape[0]

    V = np.zeros((1,1,N_s))
    for _ in trange(max_steps, desc='Value Iteration'):
        tx_r = np.multiply(tx_mat, r_mat+gamma*V)
        V_new = np.max(np.sum(tx_r, axis=2), axis=1)
        if np.max(np.abs(V_new - V)) < delta:
            break
        V = V_new

    policy_arr = np.argmax(np.sum(tx_r, axis=2), axis=1)

    return policy_arr, V

def sample_trajectory(env, seed=None):
    obs, _ = env.reset(seed=seed)
    while True:
        action = env.action_space.sample()
        new_obs, reward, terminated, truncated, _ = env.step(action)
        print(f'State: {obs}, Action: {action}, Reward: {reward}')
        obs = new_obs
        if terminated:
            print('Patient survived!' if reward > 0 else 'Patient died!')
            break
        if truncated:
            print('Episode truncated!')
            break

def estimate_j_pi(env:gym.Env, policy:Callable[[int], int], /, *, 
        num_episodes:int=50_000, gamma:float=None,
        desc:str=None
    )->tuple[float, dict[str, float]]:
    """Estimate the value of a policy with the option to specify the discount factor other
    than the one specified in the environment. Return the average value and the standard deviation."""
    gamma = env.unwrapped.gamma if gamma is None else gamma

    J_pi_arr = np.zeros(num_episodes)
    ep_lens = np.zeros(num_episodes, dtype=int)
    for i in trange(num_episodes, desc=desc):
        J_pi = 0.
        discount = 1.
        obs, _ = env.reset()
        num_steps = 0
        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            J_pi += reward * discount
            discount *= gamma
            num_steps += 1
            if terminated or truncated:
                break
        J_pi_arr[i] = J_pi
        ep_lens[i] = num_steps

    info_dict = {
        'std': np.std(J_pi_arr),
        'stderr': np.std(J_pi_arr) / np.sqrt(num_episodes),
        'min': np.min(J_pi_arr),
        'max': np.max(J_pi_arr),
        'median': np.median(J_pi_arr),
        'mean_ep_len': np.mean(ep_lens),
    }

    return np.mean(J_pi_arr), info_dict