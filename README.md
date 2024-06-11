# The ICU-Sepsis Environment

The **ICU-Sepsis** environment is a reinforcement learning environment that
simulates the treatment of sepsis in an intensive care unit (ICU). The
environment is introduced in the paper
[ICU-Sepsis: A Benchmark MDP Built from Real Medical Data](https://arxiv.org/abs/2406.05646), accepted at the 
Reinforcement Learning Conference, 2024. ICU-Sepsis is built using 
the [MIMIC-III Dataset](https://physionet.org/content/mimiciii/1.4/),
based on the work of
[Komorowski et al. (2018)](https://www.nature.com/articles/s41591-018-0213-5).

![ICU-Sepsis Environment](assets/sepsis-fig-timeline.png)

Citation:
```bibtex
@inproceedings{
choudhary2024icusepsis,
title={{ICU-Sepsis}: A Benchmark {MDP} Built from Real Medical Data},
author={Kartik Choudhary and Dhawal Gupta and Philip S. Thomas},
booktitle={Reinforcement Learning Conference},
year={2024},
}
```

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
3. **Optimal:** Optimal policy computed using value iteration (requires knowledge of the transition parameters).

## Model parameters

In addition to the Python implementation of the environment using the [Gym](https://www.gymlibrary.dev/) and
[Gymnasium](https://gymnasium.farama.org/) libraries (see below), the model parameters are also provided as CSV files
in the `icu-sepsis-csv-tables.tar.gz` archive which contains the following files:

1. `transitionFunction.csv`: The transition table with $N_S \times N_A$ rows and $N_S$ columns. The transition probability $p(s_i,a,s_f)$
   is given in the $s_f^{\text{th}}$ column of the $(s_i \times N_A + a)^{\text{th}}$ row.
2. `rewardFunction.csv`: The reward table with $1$ row and $N_S$ columns. Since the reward after any transition is a deterministic function
   of the state being transitioned into, the reward $R(s_i,a,s_f)$ is present in the $s_f^{\text{th}}$ column.
3. `initialStateDistribution.csv`: The initial-state distribution table with $1$ row and $N_S$ columns. The initial-state probability $d_0(s)$
   of the state $s$ is present in the $s^{\text{th}}$ column.
4. `expertPolicy.csv`: The expert policy table with $N_S$ rows and $N_A$ columns containing the estimated policy $\pi_{\text{expert}}$ used
   by the clinicians. The probability $\pi_{\text{expert}}(s, a)$ is present in the $a^{\text{th}}$ column of the $s^{\text{th}}$ row.

## Python installation and quickstart

ICU-Sepsis can be used with Python `3.10` or later, with gymnasium `0.28.1` or
later, and gym `0.21.0` or later.


### Installation with pip

The environment can be installed using the `pip` command:

```bash
pip install icu-sepsis
```

### Installation from source

To install the environment from source, clone the repository and navigate to
the `packages` directory, and install the `icu_sepsis` package locally:

```bash
git clone https://github.com/icu-sepsis/icu-sepsis.git
cd icu-sepsis/packages/
pip install icu_sepsis/
```

### Uninstallation

To uninstall, use the `pip uninstall` command:

```bash
pip uninstall icu_sepsis -y
```

### Quickstart

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

You can run the script `examples/quickstart.py` to verify that the
installation was successful.

## Reproducing the environment parameters

The optional helper library can be installed locally by cloning the repository,
and installing the `icu_sepsis_helpers` package in the `packages` directory:

```bash
git clone https://github.com/icu-sepsis/icu-sepsis.git
cd icu-sepsis/packages/
pip install icu_sepsis_helpers/
```

This library provides a set of helper functions to re-create the environment
parameters from scratch using the 
[MIMIC-III Dataset](https://physionet.org/content/mimiciii/1.4/) and the scripts
by
[Komorowski et al. (2018)](https://github.com/matthieukomorowski/AI_Clinician).

First, the data extraction and identification of the sepsis cohort is done
using the `AIClinician_Data_extract_MIMIC3_140219.ipynb` and
`AIClinician_sepsis3_def_160219.m` scripts from the
[AI Clinician repository](https://github.com/matthieukomorowski/AI_Clinician).

Then, the `build.py` script in the `icu_sepsis_helpers` package can be used to
create the environment parameters using the created sepsis cohort.

For convenience, the `examples/build_mimic_demo.py` script can be used to
create the environment parameters and save them to disk.

Baselines can be computed using the `examples/get_baselines.py` script.
