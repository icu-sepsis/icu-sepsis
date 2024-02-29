from gym.envs.registration import register as register_gym
from gymnasium.envs.registration import register
from icu_sepsis.envs import ICUSepsisEnv
from icu_sepsis.wrappers import (FlattenActionWrapper, GymWrapper,
                                 LegacyFlattenActionWrapper)

MAX_EPISODE_STEPS = 500


def icu_sepsis_flat(**kwargs):
    env = ICUSepsisEnv(**kwargs)
    return FlattenActionWrapper(env)


def icu_sepsis_flat_legacy(**kwargs):
    env = ICUSepsisEnv(**kwargs)
    env_legacy = GymWrapper(env)
    return LegacyFlattenActionWrapper(env_legacy)


register(
    id='Sepsis/ICU-Sepsis-v1',
    entry_point='icu_sepsis:icu_sepsis_flat',
    max_episode_steps=MAX_EPISODE_STEPS)

register(
    id='Sepsis/ICU-Sepsis-v1-cts',
    entry_point='icu_sepsis:ICUSepsisEnv',
    max_episode_steps=MAX_EPISODE_STEPS)

register_gym(
    id='Sepsis/ICU-Sepsis-v1',
    entry_point='icu_sepsis:icu_sepsis_flat_legacy',
    max_episode_steps=MAX_EPISODE_STEPS)
