"""ICU Sepsis environment registration."""

from warnings import warn

from gym.envs.registration import register as register_gym
from gymnasium.envs.registration import register

from .envs import ICUSepsisEnv
from .wrappers import (FlattenActionWrapper, GymWrapper,
                       LegacyFlattenActionWrapper)
from .utils.constants import (inadmissible_action_strategies as ias,
                              MAX_EPISODE_STEPS)


def icu_sepsis_flat(**kwargs):
    """ICU Sepsis environment with flattened action space."""
    env = ICUSepsisEnv(**kwargs)
    return FlattenActionWrapper(env)


def icu_sepsis_flat_legacy(**kwargs):
    """ICU Sepsis environment for legacy gym with flattened action space."""
    env = ICUSepsisEnv(**kwargs)
    env_legacy = GymWrapper(env)
    return LegacyFlattenActionWrapper(env_legacy)


def icu_sepsis_flat_v1(**kwargs):
    """Version 1 of the ICU Sepsis environment, where the inadmissible action
    strategy is always 'mean'. This is the version used in the paper."""

    default = ias.DEFAULT_INADMISSIBLE_ACTION_STRATEGY_V1
    if kwargs.get('inadmissible_action_strategy', default) != default:
        warn(f'Changing inadmissible_action_strategy is not supported in '
             'V1 of the ICU Sepsis environment. Please use V2 or later. '
             f'Resetting to default V1 behavior of "{default}".')
    kwargs.update(inadmissible_action_strategy=default)
    return icu_sepsis_flat(**kwargs)


def icu_sepsis_flat_v1_legacy(**kwargs):
    """Version 1 of the ICU Sepsis environment with legacy gym interface."""

    default = ias.DEFAULT_INADMISSIBLE_ACTION_STRATEGY_V1
    if kwargs.get('inadmissible_action_strategy', default) != default:
        warn(f'Changing inadmissible_action_strategy is not supported in '
             'V1 of the ICU Sepsis environment. Please use V2 or later. '
             f'Resetting to default V1 behavior of "{default}".')
    kwargs.update(inadmissible_action_strategy=default)
    return icu_sepsis_flat_legacy(**kwargs)


register(
    id='Sepsis/ICU-Sepsis-v1',
    entry_point='icu_sepsis:icu_sepsis_flat_v1',
    max_episode_steps=MAX_EPISODE_STEPS)

register_gym(
    id='Sepsis/ICU-Sepsis-v1',
    entry_point='icu_sepsis:icu_sepsis_flat_v1_legacy',
    max_episode_steps=MAX_EPISODE_STEPS)

register(
    id='Sepsis/ICU-Sepsis-v2',
    entry_point='icu_sepsis:icu_sepsis_flat',
    max_episode_steps=MAX_EPISODE_STEPS)

register_gym(
    id='Sepsis/ICU-Sepsis-v2-legacy',
    entry_point='icu_sepsis:icu_sepsis_flat_legacy',
    max_episode_steps=MAX_EPISODE_STEPS)
