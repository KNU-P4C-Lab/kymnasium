from enum import IntEnum
from typing import Literal
import numpy as np
import pygame
import pygame.freetype
from itertools import product
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball, WorldObj
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym

MISSION_NAME = 'Grid Survivor!'

SYM_AGENT = 'A'
SYM_START = 'S'
SYM_EMPTY = 'E'
SYM_WALL = 'W'
SYM_BALL = 'B'
SYM_HONEY_BEE = 'B'
SYM_HORNET = 'H'
SYM_KILLER_BEE = 'K'

DIR_RIGHT = 'R'
DIR_DOWN = 'D'
DIR_LEFT = 'L'
DIR_UP = 'U'

DIR_TO_VEC = [
    np.array((-1, 0)),
    np.array((1, 0)),
    np.array((0, -1)),
    np.array((0, 1)),
    np.array((0, 0)),
]

class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2


class HoneyBee(Ball):
    def __init__(self):
        super().__init__(color='yellow')

    def can_overlap(self) -> bool:
        return True


class Hornet(Ball):
    def __init__(self):
        super().__init__(color='blue')

    def can_overlap(self) -> bool:
        return True


class KillerBee(Ball):
    def __init__(self):
        super().__init__(color='purple')

    def can_overlap(self) -> bool:
        return True


class GridSurvivorEnv(MiniGridEnv):
    """
   Actions available:
   - 0: turn left
   - 1: turn right
   - 2: move forward
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(self,
                 max_steps: int,
                 blueprint: np.ndarray,
                 max_hit_points: int,
                 damage: int,
                 render_mode: str,
                 **kwargs):
        width, height = blueprint.shape
        start_pos = np.argwhere(blueprint == SYM_START).ravel()
        assert len(start_pos) == 2, 'Only one start position should be provided.'

        super().__init__(
            mission_space=MissionSpace(lambda: MISSION_NAME),
            width=width,
            height=height,
            max_steps=max_steps,
            **kwargs
        )
        self._blueprint = blueprint
        self._honey_bees = []
        self._hornets = []
        self._killer_bees = []
        self._max_hit_points = self._hit_points = max_hit_points
        self._damage = damage

        self.agent_start_pos = tuple(start_pos)
        self.agent_start_dir = 0
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(Actions))

    @property
    def steps_(self):
        return self.step_count

    @property
    def hit_points_(self):
        return self._hit_points

    @property
    def n_honey_bees_(self):
        return len(self._honey_bees)

    @property
    def n_hornets_(self):
        return len(self._hornets)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        for x, y in product(range(width), range(height)):
            sym = self._blueprint[x, y]
            sym_key = sym[0] if len(sym) > 1 else sym

            if sym_key == SYM_WALL:
                self.grid.set(x, y, Wall())
            elif sym_key == SYM_HONEY_BEE:
                obj = HoneyBee()
                self.put_obj(obj, x, y)
                self._honey_bees.append(obj)
            elif sym_key == SYM_HORNET:
                obj = Hornet()
                self.put_obj(obj, x, y)
                self._hornets.append(obj)
            elif sym_key == SYM_KILLER_BEE:
                obj = KillerBee()
                self.put_obj(obj, x, y)
                self._killer_bees.append(obj)

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def _reward(self) -> float:
        return 0

    def _place_object(self, obj: WorldObj, max_tries: int, positions: np.ndarray, weights: np.ndarray = None):
        if np.array_equal(obj.cur_pos, self.front_pos):
            return

        num_tries = 0
        is_same_pos = False
        probs = weights if weights is not None else np.ones(len(positions))
        probs = probs / np.sum(probs)

        while True:
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")
            num_tries += 1
            pos = self.np_random.choice(positions, p=probs)

            if np.array_equal(obj.cur_pos, pos):
                is_same_pos = True
                break

            if self.grid.get(*pos) is not None:
                continue

            if np.array_equal(pos, self.agent_pos):
                continue

            break

        if not is_same_pos:
            self.grid.set(pos[0], pos[1], obj)
            self.grid.set(obj.cur_pos[0], obj.cur_pos[1], None)
            if obj is not None:
                obj.cur_pos = pos

    def _move_object(self, obj: WorldObj, max_tries: int, mode: Literal['toward', 'random', 'stand']):
        ox, oy = obj.cur_pos
        positions = []

        for i in range(len(DIR_TO_VEC)):
            dx, dy = DIR_TO_VEC[i]
            nx, ny = ox + dx, oy + dy
            positions.append((nx, ny))

        positions = np.asarray(positions)
        weights = np.ones(len(positions))

        if mode == 'toward':
            ax, ay = self.agent_pos
            dist = np.array([np.abs(x - ax) + np.abs(y - ay) for x, y in positions])
            min_dist = np.min(dist)
            if min_dist < 10:
                weights[dist == min_dist] = 30
        elif mode == 'stand':
            weights[-1] = 10

        try:
            self._place_object(obj, max_tries, positions, weights)
        except RecursionError:
            pass

    def _gen_info(self):
        return {
            'steps': self.steps_,
            'hit_points': self.hit_points_,
            'n_honey_bees': self.n_honey_bees_,
            'n_hornets': self.n_hornets_,
        }

    def reset(self, **kwargs):
        self._hit_points = self._max_hit_points
        self._honey_bees.clear()
        self._hornets.clear()
        self._killer_bees.clear()
        obs, _ = super().reset(**kwargs)
        return obs, self._gen_info()

    def step(self, action):
        is_game_over = False

        if action == self.actions.forward:
            for obj in self._hornets:
                self._move_object(obj, 100, 'random')
            for obj in self._killer_bees:
                self._move_object(obj, 100, 'toward')
            for obj in self._honey_bees:
                self._move_object(obj, 100, 'stand')

            front_obj = self.grid.get(*self.front_pos)
            is_saved = front_obj and front_obj.type == 'ball' and front_obj.color == 'yellow'
            is_attacked = front_obj and front_obj.type == 'ball' and front_obj.color == 'blue'
            is_killed = front_obj and front_obj.type == 'ball' and front_obj.color == 'purple'

            if is_saved:
                self.grid.set(front_obj.cur_pos[0], front_obj.cur_pos[1], None)
                self._honey_bees.remove(front_obj)
                is_game_over = len(self._honey_bees) <= 0
            elif is_attacked:
                self.grid.set(front_obj.cur_pos[0], front_obj.cur_pos[1], None)
                self._hornets.remove(front_obj)
                self._hit_points = max(0, self._hit_points - self._damage)
                is_game_over = self._hit_points <= 0
            elif is_killed:
                is_game_over = True

        obs, reward, terminated, truncated, _ = super().step(action)

        if is_game_over:
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
        text = f'Steps: {self.steps_} / Hit Points: {self.hit_points_} / Honey Bees: {self.n_honey_bees_} / Hornets: {self.n_hornets_}'
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


