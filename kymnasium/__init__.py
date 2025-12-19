"""
kymnasium - A reinforcement learning environment package

This package provides custom environments for reinforcement learning experiments.
"""

import os
import gymnasium as gym
from .agent import Agent
from .evaluate import evaluate, evaluate_remote, RemoteEnvWrapper, InvalidActionError, NotAllowedUserIdError
from . import alkkagi, mario_world, grid_world, grid_survivor


__all__ = [
    # Core classes and methods
    'Agent',
    'evaluate',
    'evaluate_remote',
    'RemoteEnvWrapper',
    'InvalidActionError',
    'NotAllowedUserIdError',
    # Modules
    'alkkagi',
    'avoid_blurp',
    'grid_adventure',
    'grid_survivor'
]

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

__version__ = "1.1.1"



# Al-Kka-Gi
# -------------------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------------------


# Avoid Blurp
# -------------------------------------------------------------------------------------------

gym.register(
    id='kymnasium/AvoidBlurp-Easy-v0',
    entry_point='kymnasium.mario_world.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.5,
        max_spawns=30,
        prob_spawn_on_player=0.0,
        max_spawn_duration=105
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Normal-v0',
    entry_point='kymnasium.mario_world.registration:_create_env',
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
    entry_point='kymnasium.mario_world.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.1,
        max_spawns=30,
        prob_spawn_on_player=0.2,
        max_spawn_duration=105
    )
)
# -------------------------------------------------------------------------------------------


# Grid Adventure
# -------------------------------------------------------------------------------------------
gym.register(
    id='kymnasium/GridAdventure-FullMaze-26x26-v0',
    entry_point='kymnasium.grid_world.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=1000,
        blueprint=os.path.join(
            os.path.dirname(__file__),
            'grid_world',
            'assets',
            'full-maze-26x26-v0.csv'
        ),
    )
)

gym.register(
    id='kymnasium/GridAdventure-FullMaze-32x32-v0',
    entry_point='kymnasium.grid_world.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=1000,
        blueprint=os.path.join(
            os.path.dirname(__file__),
            'grid_world',
            'assets',
            'full-maze-32x32-v0.csv'
        ),
    )
)

gym.register(
    id='kymnasium/GridAdventure-Crossing-26x26-v0',
    entry_point='kymnasium.grid_world.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=1000,
        blueprint=os.path.join(
            os.path.dirname(__file__),
            'grid_world',
            'assets',
            'crossing-26x26-v0.csv'
        ),
    )
)


# Grid Survivor
# -------------------------------------------------------------------------------------------
gym.register(
    id='kymnasium/GridSurvivor-Rescue-34x34-v0',
    entry_point='kymnasium.grid_survivor.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=1000,
        blueprint=os.path.join(
            os.path.dirname(__file__),
            'grid_survivor',
            'assets',
            'rescue-34x34-v0.csv'
        ),
        max_hit_points=100,
        damage=20
    )
)