import os
import gymnasium as gym
from .wrappers import GridSurvivorManualPlayWrapper as ManualPlayWrapper


__all__ = [
    'ManualPlayWrapper'
]


gym.register(
    id='kymnasium/GridSurvivor-Rescue-34x34-v0',
    entry_point='kymnasium.grid_survivor.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=1000,
        blueprint=os.path.join(os.path.dirname(__file__), 'assets', 'rescue-34x34-v0.csv'),
        max_hit_points=100,
        damage=20
    )
)