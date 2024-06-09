# The ICU-Sepsis Environment

The **ICU-Sepsis** environment is a reinforcement learning environment that
simulates the treatment of sepsis in an intensive care unit (ICU).

## Environment description

ICU-Sepsis is a tabular MDP with $N_S = 716$ states ($\\{0,1,\dots,715\\}$) and $N_A = 25$ actions ($\\{0,1,\dots,24\\}$).
Each episode simulates the treatment of one sepsis patient in the ICU.

An episode ends when the patient survives (state $714$) and gets a reward of `+1`, or dies (state $713$) and gets a reward
of `0`, while all the intermediate rewards are `0`. The discount factor is set as $\gamma = 1$.

## Baselines

Some baseline results are shown below as a reference.

<table>
  <tr>
    <th></th>
    <th>Random</th>
    <th>Expert</th>
    <th>Optimal</th>
  </tr>
  <tr>
    <th>Avg. return</th>
    <td>0.78</td>
    <td>0.78</td>
    <td>0.88</td>
  </tr>
  <tr>
    <th>Avg. episode length</th>
    <td>9.45</td>
    <td>9.22</td>
    <td>10.99</td>
  </tr>
</table>

The three baseline policies used are:
1. **Random:** Each action is taken uniformly randomly out of all the actions in any given state.
2. **Expert:** The estimated policy used by clinicians in the real world, computed using the data from the MIMIC-III dataset.
3. **Optimal:** Optimal policy computed using value iteration (requires knowledge of the transition parameters)

## Installation

ICU-Sepsis can be used with Python `3.10` or later, with gymnasium `0.28.1` or
later, and gym `0.21.0` or later. The environment can be installed using
the `pip` command:

```bash
pip install icu-sepsis
```

### Uninstallation

To uninstall, use the `pip uninstall` command:

```bash
pip uninstall icu_sepsis -y
```

## Quickstart

The environment can be loaded with the Gym or Gymnasium packages and follows
the standard Gym API. The following code snippet demonstrates how to create
the environment, reset it, and take a step:

```python
import gymnasium as gym
import icu_sepsis

env = gym.make('Sepsis/ICU-Sepsis-v1')

state, info = env.reset()
print('Initial state:', state)

next_state, reward, terminated, truncated, info = env.step(0)
print('Next state:', next_state)
print('Reward:', reward)
print('Terminated:', terminated)
print('Truncated:', truncated)
```
