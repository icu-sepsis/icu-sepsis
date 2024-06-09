# The ICU-Sepsis Helper Library

The **ICU-Sepsis** environment is a reinforcement learning environment that
simulates the treatment of sepsis in an intensive care unit (ICU).

The helper library `icu_sepsis_helpers` can be installed locally by cloning
the ICU-Sepsis repository, and installing the `icu_sepsis_helpers` package in
the `packages` directory:

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
