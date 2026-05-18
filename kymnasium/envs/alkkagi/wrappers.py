import gymnasium as gym
import numpy as np
from typing import Any
import pygame
import pygame.math
import math
from .objs import SlingShot
from .env import AlkkagiEnv
from kymnasium.common.evaluate import RemoteEnvWrapper
from kymnasium.common.agent import Agent
from kymnasium.common.manual import ManualPlayWrapper


class AlkkagiManualPlayWrapper(ManualPlayWrapper):
    def __init__(self, env_id: str, debug: bool = False, agent: Agent = None, agent_turn: int = None, **kwargs):
        super().__init__(env_id, debug, **kwargs)

        self._is_dragging = False
        self._pos_start = None
        self._pos_end = None
        self._selected_stone = None
        self._agent_turn = agent_turn
        self._agent = agent

    def handle_events(self, event):
        env = self.env.unwrapped
        assert isinstance(env, AlkkagiEnv)

        if self._agent_turn is not None and self._agent is not None:
            if self._agent_turn == env.turn_:
                return self._agent.act(env.latest_obs_, env.latest_info_)

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
            power = min(distance * env.max_power / SlingShot.POWER_LINE_LENGTH, env.max_power)
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
        env = env.unwrapped
        assert isinstance(env, AlkkagiEnv)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0, high=255, shape=(env.width, env.height, 3), dtype=np.uint8
            ),
            'turn': gym.spaces.Discrete(2),
        })

    def observation(self, observation):
        env = self.env.unwrapped
        assert isinstance(env, AlkkagiEnv)
        img = env.get_frame()
        return {
            'image': img,
            'turn': env.turn_,
        }


class AlkkagiRemoteEnvWrapper(RemoteEnvWrapper):
    def serialize(self, observation: Any):
        return observation

    def verify_action(self, action) -> bool:
        env = self.env.unwrapped
        return isinstance(env, AlkkagiEnv) and action['turn'] == env.turn_
