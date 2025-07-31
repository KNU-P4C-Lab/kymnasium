import gymnasium as gym
"""
kymnasium - A reinforcement learning environment package

This package provides custom environments for reinforcement learning experiments.
"""

__version__ = "0.1.0"

from kymnasium import grid_adventure
from kymnasium import alkkagi
from kymnasium import avoid_blurp
from .agent import Agent
from .evaluate import LocalEvaluator, RemoteEvaluator, RemoteEnvWrapper, InvalidActionError, InvalidIdError

__all__ = [
    'Agent',
    'LocalEvaluator',
    'RemoteEvaluator',
    'RemoteEnvWrapper',
    'InvalidActionError',
    'InvalidIdError'
]