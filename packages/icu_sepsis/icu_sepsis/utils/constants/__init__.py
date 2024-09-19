"""Constants used in the icu-sepsis package."""

# Environment constants
MAX_EPISODE_STEPS = 500

# State meanings
STATE_DEATH = 713
STATE_SURVIVAL = 714
STATE_S_INF = 715
STATES_TERMINAL = {STATE_DEATH, STATE_SURVIVAL, STATE_S_INF}
