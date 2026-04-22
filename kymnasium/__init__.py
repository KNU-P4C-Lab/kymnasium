"""
kymnasium - A reinforcement learning environment package

This package provides custom environments for reinforcement learning experiments.
"""

import os
import gymnasium as gym
from .agent import Agent
from .evaluate import evaluate, evaluate_remote, RemoteEnvWrapper, InvalidActionError, NotAllowedUserIdError
from . import alkkagi, avoid_blurp, grid_world, bullet_bill, zelda_adventure


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
    'grid_world',
    'bullet_bill',
    'zelda_adventure',
]

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

__version__ = "1.2.3"



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
    id='kymnasium/AvoidBlurp-Discrete-Vertical-Easy-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.5,
        max_spawns=30,
        prob_spawn_on_player=0.0,
        max_spawn_duration=105,
        continuous_action=False,
        mode='vertical',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Discrete-Vertical-Normal-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.3,
        max_spawns=30,
        prob_spawn_on_player=0.1,
        max_spawn_duration=105,
        continuous_action=False,
        mode='vertical',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Discrete-Vertical-Hard-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.1,
        max_spawns=30,
        prob_spawn_on_player=0.2,
        max_spawn_duration=105,
        continuous_action=False,
        mode='vertical',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Continuous-Vertical-Easy-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.5,
        max_spawns=30,
        prob_spawn_on_player=0.0,
        max_spawn_duration=105,
        continuous_action=True,
        mode='vertical',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Continuous-Vertical-Normal-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.3,
        max_spawns=30,
        prob_spawn_on_player=0.1,
        max_spawn_duration=105,
        continuous_action=True,
        mode='vertical',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Continuous-Vertical-Hard-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.1,
        max_spawns=30,
        prob_spawn_on_player=0.2,
        max_spawn_duration=105,
        continuous_action=True,
        mode='vertical',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Discrete-Ballistic-Easy-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.5,
        max_spawns=30,
        prob_spawn_on_player=0.0,
        max_spawn_duration=105,
        continuous_action=False,
        mode='ballistic',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.3,
        max_spawns=30,
        prob_spawn_on_player=0.1,
        max_spawn_duration=105,
        continuous_action=False,
        mode='ballistic',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Discrete-Ballistic-Hard-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.1,
        max_spawns=30,
        prob_spawn_on_player=0.2,
        max_spawn_duration=105,
        continuous_action=False,
        mode='ballistic',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Continuous-Ballistic-Easy-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.5,
        max_spawns=30,
        prob_spawn_on_player=0.0,
        max_spawn_duration=105,
        continuous_action=True,
        mode='ballistic',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Continuous-Ballistic-Normal-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.3,
        max_spawns=30,
        prob_spawn_on_player=0.1,
        max_spawn_duration=105,
        continuous_action=True,
        mode='ballistic',
        stage=1
    )
)

gym.register(
    id='kymnasium/AvoidBlurp-Continuous-Ballistic-Hard-Stage-1',
    entry_point='kymnasium.avoid_blurp.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.1,
        max_spawns=30,
        prob_spawn_on_player=0.2,
        max_spawn_duration=105,
        continuous_action=True,
        mode='ballistic',
        stage=1
    )
)
# -------------------------------------------------------------------------------------------


# Bullet Bill
# -------------------------------------------------------------------------------------------

gym.register(
    id='kymnasium/BulletBill-Discrete-Easy-Stage-1',
    entry_point='kymnasium.bullet_bill.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.5,
        max_spawns=30,
        max_spawn_duration=105,
        continuous_action=False,
    )
)

gym.register(
    id='kymnasium/BulletBill-Discrete-Normal-Stage-1',
    entry_point='kymnasium.bullet_bill.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.3,
        max_spawns=30,
        max_spawn_duration=105,
        continuous_action=False,
    )
)

gym.register(
    id='kymnasium/BulletBill-Discrete-Hard-Stage-1',
    entry_point='kymnasium.bullet_bill.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.1,
        max_spawns=30,
        max_spawn_duration=105,
        continuous_action=False,
    )
)

gym.register(
    id='kymnasium/BulletBill-Continuous-Easy-Stage-1',
    entry_point='kymnasium.bullet_bill.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.5,
        max_spawns=30,
        max_spawn_duration=105,
        continuous_action=True,
    )
)

gym.register(
    id='kymnasium/BulletBill-Continuous-Normal-Stage-1',
    entry_point='kymnasium.bullet_bill.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.3,
        max_spawns=30,
        max_spawn_duration=105,
        continuous_action=True,
    )
)

gym.register(
    id='kymnasium/BulletBill-Continuous-Hard-Stage-1',
    entry_point='kymnasium.bullet_bill.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        game_duration=120,
        init_spawn_interval=1.5,
        min_spawn_interval=0.1,
        max_spawns=30,
        max_spawn_duration=105,
        continuous_action=True,
    )
)

# Grid World
# -------------------------------------------------------------------------------------------
gym.register(
    id='kymnasium/GridWorld-Adventure-26x26',
    entry_point='kymnasium.grid_world.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=500,
        blueprint=os.path.join(
            os.path.dirname(__file__),
            'grid_world',
            'assets',
            'adventure-26x26.csv'
        ),
    )
)

gym.register(
    id='kymnasium/GridWorld-Adventure-32x32',
    entry_point='kymnasium.grid_world.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=500,
        blueprint=os.path.join(
            os.path.dirname(__file__),
            'grid_world',
            'assets',
            'adventure-32x32.csv'
        ),
    )
)

gym.register(
    id='kymnasium/GridWorld-Crossing-26x26',
    entry_point='kymnasium.grid_world.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=500,
        blueprint=os.path.join(
            os.path.dirname(__file__),
            'grid_world',
            'assets',
            'crossing-26x26.csv'
        ),
    )
)


# Zelda Adventure
# -------------------------------------------------------------------------------------------
gym.register(
    id='kymnasium/ZeldaAdventure-Stage-1',
    entry_point='kymnasium.zelda_adventure.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=1000,
        stage=1
    )
)

gym.register(
    id='kymnasium/ZeldaAdventure-Stage-2',
    entry_point='kymnasium.zelda_adventure.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=1000,
        stage=2
    )
)

gym.register(
    id='kymnasium/ZeldaAdventure-Stage-3',
    entry_point='kymnasium.zelda_adventure.registration:_create_env',
    disable_env_checker=True,
    kwargs=dict(
        max_steps=1000,
        stage=3
    )
)
