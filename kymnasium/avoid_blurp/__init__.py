import gymnasium as gym
from .wrappers import AvoidBlurpManualWrapper as ManualPlayWrapper


__all__ = [
    'ManualPlayWrapper'
]


gym.register(
    id='kymnasium/AvoidBlurp-Easy-v0',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=2.0,
        min_spawn_interval=0.5,
        max_spawns=30,
        prob_spawn_on_player=0.0,
        max_spawn_duration=120
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Normal-v0',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.3,
        max_spawns=30,
        prob_spawn_on_player=0.1,
        max_spawn_duration=105
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Hard-v0',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.0,
        min_spawn_interval=0.1,
        max_spawns=30,
        prob_spawn_on_player=0.2,
        max_spawn_duration=90
    )
)