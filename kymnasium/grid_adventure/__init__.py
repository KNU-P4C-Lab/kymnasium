import os
import gymnasium as gym
from .wrappers import GridAdventureManualPlayWrapper as ManualPlayWrapper

__all__ = [
    'ManualPlayWrapper'
]

gym.register(
    id='kymnasium/GridAdventure-FullMaze-26x26-v0',
    entry_point='kymnasium.grid_adventure.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=500,
        blueprint=os.path.join(os.path.dirname(__file__), 'assets', 'full-maze-26x26-v0.csv'),
    )
)

gym.register(
    id='kymnasium/GridAdventure-Crossing-26x26-v0',
    entry_point='kymnasium.grid_adventure.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=500,
        path=os.path.join(os.path.dirname(__file__), 'assets', 'crossing-26x26-v0.csv'),
    )
)