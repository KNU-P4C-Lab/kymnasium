import gymnasium as gym
import numpy as np
import pygame
import pygame.math
import math
from .env import AlkkagiEnv, MAX_POWER, POWER_LINE_LENGTH, GAME_HEIGHT, GAME_WIDTH
from ..util import ManualPlayWrapper


class AlkkagiManualPlayWrapper(ManualPlayWrapper):
    def __init__(self, env_id: str, debug: bool = False, **kwargs):
        super().__init__(env_id, debug, **kwargs)

        self._is_dragging = False
        self._pos_start = None
        self._pos_end = None
        self._selected_stone = None

    def handle_events(self, event):
        env = self.env.unwrapped

        if not isinstance(env, AlkkagiEnv):
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self._is_dragging:
            mouse_pos = pygame.mouse.get_pos()
            if not self._is_dragging:
                self._selected_stone = env.find_stone(mouse_pos)
                if self._selected_stone is not None:
                    self._is_dragging = True
                    self._pos_start = self._selected_stone.position_
                    self._pos_end = mouse_pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._is_dragging:
            sx, sy = self._pos_start
            ex, ey = self._pos_end
            dx, dy = sx - ex, sy - ey
            angle = pygame.math.Vector2(1, 0).angle_to(pygame.math.Vector2(dx, dy))
            distance = math.hypot(dx, dy)
            power = min(distance * MAX_POWER / POWER_LINE_LENGTH, MAX_POWER)
            action = {
                'turn': env.turn_,
                'angle': angle,
                'power': power,
                'index': self._selected_stone.index_
            }
            self._is_dragging = False
            self._pos_start = None
            self._pos_end = None
            self._selected_stone = None
            return action
        elif event.type == pygame.MOUSEMOTION and self._is_dragging:
            self._pos_end = pygame.mouse.get_pos()

        env.set_slingshot_from_pos(self._pos_start, self._pos_end)

        return None


class RGBImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8),
            'current_player': gym.spaces.Discrete(2),
        })

    def observation(self, observation):
        env = self.env.unwrapped
        if isinstance(env, AlkkagiEnv):
            img = env.get_frame()

            return {
                'image': img,
                'turn': env.turn_,
            }
        return None

