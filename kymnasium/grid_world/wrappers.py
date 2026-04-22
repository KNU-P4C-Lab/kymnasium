import numpy as np
import gymnasium as gym
import pygame
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper as MiniGridRGBImgObsWrapper, \
    FullyObsWrapper as MiniGridFullyObsWrapper
from itertools import product
from .env import Actions
from ..manual import ManualPlayWrapper


class FullyObsWrapper(MiniGridFullyObsWrapper):
    """
    (width, height, 3) numpy array observation wrapper for Grid Adventure,
    where each pixel is an encoding of the object type, color, and state.
    object type: 1=empty, 2=wall, 4=door, 5=key, 6=ball, 8=goal, 9=lava, 10=agent
    color: 0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=gray only for a door, a key, and a ball.
    state: 0=open, 1=closed, 2=locked for a door; 0=right, 1=down, 2=left, 3=up for an agent
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        env_unwrapped = env.unwrapped

        assert isinstance(env_unwrapped, MiniGridEnv)
        width, height = env_unwrapped.width, env_unwrapped.height

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1600,
            shape=(width, height, ),
            dtype=np.uint16,
        )

    def observation(self, obs):
        obs = super().observation(obs)
        img = obs['image']
        row, col = img.shape[:2]
        arr = np.zeros((row, col), dtype=np.uint16)
        for i, j in product(range(row), range(col)):
            arr[j, i] = int(''.join(map(str, img[i, j])))
        return arr


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


class GridAdventureManualPlayWrapper(ManualPlayWrapper):
    KEY_TO_ACTION = {
        pygame.K_LEFT: Actions.left,
        pygame.K_RIGHT: Actions.right,
        pygame.K_UP: Actions.forward,
        pygame.K_SPACE: Actions.unlock,
        pygame.K_TAB: Actions.pickup,
        pygame.K_LSHIFT: Actions.drop,
    }

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            return self.KEY_TO_ACTION.get(event.key)
        return None


