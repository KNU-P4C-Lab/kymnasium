import pygame
from .env import ZeldaAdventureEnv
from .consts import *
from ...common.manual import ManualPlayWrapper


class ZeldaAdventureEnvManualPlayWrapper(ManualPlayWrapper):
    KEY_TO_DISCRETE_ACTION = {
        pygame.K_LEFT: Actions.turn_left,
        pygame.K_RIGHT: Actions.turn_right,
        pygame.K_UP: Actions.move_forward,
        pygame.K_q: Actions.attack,
        pygame.K_w: Actions.pick_up,
        pygame.K_e: Actions.drop,
        pygame.K_r: Actions.stay,
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
