import gymnasium as gym
import numpy as np
import pygame
from .env import GAME_WIDTH, GAME_HEIGHT, Actions, AvoidBlurpEnv
from ..util import ManualPlayWrapper


class RGBImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(GAME_WIDTH, GAME_HEIGHT, 3), dtype=np.uint8)

    def observation(self, obs):
        env = self.env.unwrapped
        if isinstance(env, AvoidBlurpEnv):
            return env.get_frame()
        return None


class AvoidBlurpManualPlayWrapper(ManualPlayWrapper):
    KEY_TO_ACTION = {
        pygame.K_LEFT: Actions.LEFT,
        pygame.K_RIGHT: Actions.RIGHT,
    }

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            return self.KEY_TO_ACTION.get(event.key, Actions.NOOP)
        return Actions.NOOP

    def default_action_(self):
        return Actions.NOOP
