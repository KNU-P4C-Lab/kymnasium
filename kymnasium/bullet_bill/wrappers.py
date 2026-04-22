import gymnasium as gym
import numpy as np
import pygame
from .env import BulletBillEnv
from .consts import *
from ..manual import ManualPlayWrapper


class RGBImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        env = env.unwrapped
        assert isinstance(env, BulletBillEnv)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(env.height, env.width, 3), dtype=np.uint8)

    def observation(self, obs):
        env = self.env.unwrapped
        assert isinstance(env, BulletBillEnv)
        return env.get_frame()


class BulletBillManualPlayWrapper(ManualPlayWrapper):
    KEY_TO_CONTINUOUS_ACTION = {
        pygame.K_LEFT: np.array([-MARIO_DEFAULT_MOVING_FORCE, 0]),
        pygame.K_RIGHT: np.array([MARIO_DEFAULT_MOVING_FORCE, 0]),
        pygame.K_SPACE: np.array([0, MARIO_JUMPING_FORCE]),
    }

    KEY_TO_DISCRETE_ACTION = {
        pygame.K_LEFT: Actions.left,
        pygame.K_RIGHT: Actions.right,
        pygame.K_SPACE: Actions.jump
    }

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, **kwargs)
        env = self.env.unwrapped
        assert isinstance(env, BulletBillEnv)
        self._continuous_action = env.continuous_action
        self._keys_to_action = self.KEY_TO_CONTINUOUS_ACTION if self._continuous_action else self.KEY_TO_DISCRETE_ACTION

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            return self._keys_to_action.get(event.key)
        return None

    @property
    def default_action_(self):
        if self._continuous_action:
            return np.array([0.0, 0.0])
        else:
            return Actions.noop
