from enum import IntEnum
import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from itertools import product
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Lava, Key, Door, Goal, Wall, Ball
from minigrid.minigrid_env import MiniGridEnv


MISSION_NAME = 'Grid Adventure!'

SYM_AGENT = 'A'
SYM_START = 'S'
SYM_GOAL = 'G'
SYM_WALL = 'W'
SYM_LAVA = 'L'
SYM_KEY = 'K'
SYM_DOOR = 'D'
SYM_BALL = 'B'
SYM_EMTPY = 'E'

GENERATABLE_OBJECTS = [
    SYM_WALL,
    SYM_LAVA,
    SYM_DOOR,
    SYM_KEY,
    SYM_BALL
]

STATE_LOCKED = 'L'
STATE_OPEN = 'O'
STATE_CLOSED = 'C'

COLOR_RED = 'R'
COLOR_BLUE = 'B'
COLOR_GREEN = 'G'
COLOR_YELLOW = 'Y'
COLOR_PURPLE = 'P'

DIR_RIGHT = 'R'
DIR_DOWN = 'D'
DIR_LEFT = 'L'
DIR_UP = 'U'

COLOR_TO_ARGUMENT = {
    COLOR_RED: 'red',
    COLOR_BLUE: 'blue',
    COLOR_GREEN: 'green',
    COLOR_YELLOW: 'yellow',
    COLOR_PURPLE: 'purple',
}


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    pickup = 3
    drop = 4
    unlock = 5


class GridWorldEnv(MiniGridEnv):
    """
    Actions available:
    - 0: turn left
    - 1: turn right
    - 2: move forward
    - 3: pick up the key
    - 4: drop the key
    - 5: unlock the door when the key is present
    """
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(self, max_steps: int, blueprint: np.ndarray, render_mode: str, **kwargs):
        width, height = blueprint.shape
        start_pos = np.argwhere(blueprint == SYM_START)[0]
        assert len(start_pos) == 2, 'Only one start position should be provided.'

        goal_pos = np.argwhere(blueprint == SYM_GOAL)[0]
        assert len(goal_pos) == 2, 'Only one goal position should be provided.'

        super().__init__(
            mission_space=MissionSpace(lambda : MISSION_NAME),
            width=width,
            height=height,
            max_steps=max_steps,
            **kwargs
        )
        self.render_mode = render_mode

        self.obstacles = []

        self.blueprint = blueprint
        self.goal_pos = goal_pos
        self.agent_start_pos = start_pos
        self.agent_start_dir = 0
        self.action_space = gym.spaces.Discrete(len(Actions))

    @property
    def steps_(self):
        return self.step_count

    @property
    def distance_(self):
        x_a, y_a = self.agent_pos
        x_g, y_g = self.goal_pos
        return abs(x_a - x_g) + abs(y_a - y_g)

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)

        for x, y in product(range(width), range(height)):
            symbol= str(self.blueprint[x, y])
            symbol_type = symbol[0] if len(symbol) > 1 else symbol

            if symbol_type not in GENERATABLE_OBJECTS:
                continue

            if symbol_type == SYM_WALL:
                self.grid.set(x, y, Wall())
            elif symbol_type == SYM_LAVA:
                self.put_obj(Lava(), x, y)
            elif symbol_type == SYM_BALL:
                obs = Ball()
                self.put_obj(obs, x, y)
                self.obstacles.append(obs)
            elif symbol_type == SYM_KEY:
                color = COLOR_TO_ARGUMENT[symbol[1]] or 'red'
                self.put_obj(Key(color=color), x, y)
            elif symbol_type == SYM_DOOR:
                color = COLOR_TO_ARGUMENT[symbol[1]] or 'red'
                is_open = False
                is_locked = False

                if symbol[2] == STATE_OPEN:
                    is_open = True
                elif symbol[2] == STATE_LOCKED:
                    is_locked = True
                self.put_obj(Door(color=color, is_open=is_open, is_locked=is_locked), x, y)

        self.put_obj(Goal(), int(self.goal_pos[0]), int(self.goal_pos[1]))
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def _reward(self) -> float:
        return 0

    def _gen_info(self):
        return {'steps': self.steps_, 'distance': self.distance_}

    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        self.render()
        return obs, self._gen_info()

    def step(self, action):
        front_obj = self.grid.get(*self.front_pos)
        is_crashed = front_obj and front_obj.type == "ball"

        for i_obst in range(len(self.obstacles)):
            x_old, y_old = self.obstacles[i_obst].cur_pos
            x_top, y_top = x_old - 1, y_old - 1

            try:
                self.place_obj(
                    self.obstacles[i_obst], top=(x_top, y_top), size=(3, 3), max_tries=100
                )
                self.grid.set(x_old, y_old, None)
            except RecursionError:
                pass

        obs, reward, terminated, truncated, _ = super().step(action)

        if action == Actions.forward and is_crashed:
            terminated = True

        return obs, reward, terminated, truncated, self._gen_info()

    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, False)
        img = np.transpose(img, axes=(1, 0, 2))

        if self.render_size is None:
            self.render_size = img.shape[:2]

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption(self.spec.id)
            else:
                self.window = pygame.Surface((self.screen_size, self.screen_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(img)
        offset = surf.get_size()[0] * 0.1
        bg = pygame.Surface(
            (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
        )
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        font_size = 22
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text = f'Steps: {self.steps_} / Dist.: {self.distance_}'
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = int(bg.get_height() - font_size * 1.5)
        font.render_to(bg, text_rect, text, size=font_size)

        self.window.blit(bg, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
        return None
