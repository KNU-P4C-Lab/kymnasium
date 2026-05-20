import numpy as np
import pygame
from .env import AvoidBlurpEnv
from .consts import *
from ...common.manual import ManualPlayWrapper


class AvoidBlurpManualPlayWrapper(ManualPlayWrapper):
    KEY_TO_CONTINUOUS_ACTION = {
        pygame.K_LEFT: np.array([-1.0, 0.0], dtype=np.float32),
        pygame.K_RIGHT: np.array([1.0, 0.0], dtype=np.float32),
        pygame.K_SPACE: np.array([0.0, 1.0], dtype=np.float32),
    }

    KEY_TO_DISCRETE_ACTION = {
        pygame.K_LEFT: Action.left_move,
        pygame.K_RIGHT: Action.right_move,
        pygame.K_SPACE: Action.jump,
    }

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        env = self.env.unwrapped
        assert isinstance(env, AvoidBlurpEnv)
        self._continuous_action = env.continuous_action
        self._keys_to_action = self.KEY_TO_CONTINUOUS_ACTION if self._continuous_action else self.KEY_TO_DISCRETE_ACTION

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            return self._keys_to_action.get(event.key)
        return None

    @property
    def default_action_(self):
        if self._continuous_action:
            return np.array([0.0, 0.0], dtype=np.float32)
        else:
            return Action.noop

