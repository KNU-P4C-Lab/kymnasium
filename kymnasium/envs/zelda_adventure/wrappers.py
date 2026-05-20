import pygame
from .env import ZeldaAdventureEnv
from .consts import *
from ...common.manual import ManualPlayWrapper


class ZeldaAdventureEnvManualPlayWrapper(ManualPlayWrapper):
    KEY_TO_DISCRETE_ACTION = {
        pygame.K_LEFT: Action.turn_left,
        pygame.K_RIGHT: Action.turn_right,
        pygame.K_UP: Action.move_forward,
        pygame.K_q: Action.attack,
        pygame.K_w: Action.pick_up,
        pygame.K_e: Action.drop,
        pygame.K_r: Action.stay,
    }

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        env = self.env.unwrapped
        assert isinstance(env, ZeldaAdventureEnv)
        self._keys_to_action = self.KEY_TO_DISCRETE_ACTION

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            return self.KEY_TO_DISCRETE_ACTION.get(event.key)
        return None

    @property
    def default_action_(self):
        return None
