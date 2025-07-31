import numpy as np
import gymnasium as gym
import pygame
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper as MiniGridRGBImgObsWrapper, \
    FullyObsWrapper as MiniGridFullyObsWrapper
from kymnasium.grid_survivor.env import Actions
from kymnasium.util import ManualPlayWrapper


class FullyObsWrapper(MiniGridFullyObsWrapper):
    """
    (width, height, 3) numpy array observation wrapper for Grid Adventure,
    where each pixel is an encoding of the object type, color, and state.
    object type: 1=empty, 2=wall, 6=ball, 10=agent
    color: 0=red, 1=green, 2=blue, 3=purple, 4=yellow only for a ball;
           honey_bee=[6, 4], hornet=[6, 2], killer_bee=[6, 3]
    state: 0=right, 1=down, 2=left, 3=up for an agent
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        env_unwrapped = env.unwrapped

        assert isinstance(env_unwrapped, MiniGridEnv)
        width, height = env_unwrapped.width, env_unwrapped.height

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(width, height, 3),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = super().observation(obs)
        return obs['image']


class RGBImgObsWrapper(MiniGridRGBImgObsWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.
    """
    def __init__(self, env: MiniGridEnv):
        super().__init__(env)

    def observation(self, obs):
        obs = super().observation(obs)
        return obs['image']


class GridSurvivorManualPlayWrapper(ManualPlayWrapper):
    KEY_TO_ACTION = {
        pygame.K_LEFT: Actions.left,
        pygame.K_RIGHT: Actions.right,
        pygame.K_UP: Actions.forward,
    }

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            return self.KEY_TO_ACTION.get(event.key)
        return None


