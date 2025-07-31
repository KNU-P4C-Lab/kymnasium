import gymnasium as gym
from .wrappers import AlkkagiManualPlayWrapper as ManualPlayWrapper
from .remote import AlkkagiRemoteEnvWrapper as RemoteEnvWrapper


__all__ = [
    'ManualPlayWrapper',
    'RemoteEnvWrapper'
]

gym.register(
    id='kymnasium/AlKkaGi-3x3-v0',
    entry_point='kymnasium.alkkagi.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        n_stones=3,
        n_obstacles=3
    )
)

gym.register(
    id='kymnasium/AlKkaGi-5x5-v0',
    entry_point='kymnasium.alkkagi.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        n_stones=5,
        n_obstacles=3
    )
)

gym.register(
    id='kymnasium/AlKkaGi-7x7-v0',
    entry_point='kymnasium.alkkagi.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        n_stones=7,
        n_obstacles=3
    )
)

gym.register(
    id='kymnasium/AlKkaGi-9x9-v0',
    entry_point='kymnasium.alkkagi.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        n_stones=9,
        n_obstacles=3
    )
)