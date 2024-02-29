# The ICU-Sepsis Environment

The ICU-Sepsis environment is a reinforcement learning environment that
simulates the treatment of sepsis in an intensive care unit (ICU). The
environment is built using the
[MIMIC-III Dataset](https://physionet.org/content/mimiciii/1.4/),
based on the work of
[Komorowski et al. (2018)](https://www.nature.com/articles/s41591-018-0213-5).

## Installation

The requirements for the environment are listed in the `requirements.txt` file, and can be installed using the `pip` command:

```bash
pip install -r requirements.txt
```

The environment can be installed locally by opening the `packages` directory in terminal and using the `pip` command:

```bash
cd packages/
pip install -e icu_sepsis/
```

To uninstall, use the `pip uninstall` command:

```bash
pip uninstall icu_sepsis -y
```

## Quick Start

The environment can be loaded with the Gym or Gymnasium packages, and follows
the standard Gym API.

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

## Optional Helper Library

The optional helper library can be installed locally by opening the `packages`
directory in terminal and using the `pip` command:

```bash
cd packages/
pip install -e icu_sepsis_helpers/
```

This library provides a set of helper functions to create the environment
parameters from scratch using the 
MIMIC-III Dataset](https://physionet.org/content/mimiciii/1.4/) and the scripts
by
[Komorowski et al. (2018)](https://github.com/matthieukomorowski/AI_Clinician).

First, the data extraction and identification of the sepsis cohort is done
using the `AIClinician_Data_extract_MIMIC3_140219.ipynb` and
`AIClinician_sepsis3_def_160219.m` scripts from the
[AI Clinician repository](https://github.com/matthieukomorowski/AI_Clinician).

Then, the following scripts `build.py` script in the `icu_sepsis_helpers`
package can be used to create the environment parameters using the created
sepsis cohort.

For convenience, the `examples/build_mimic_demo.py` script can be used to
create the environment parameters and save them to disk.
