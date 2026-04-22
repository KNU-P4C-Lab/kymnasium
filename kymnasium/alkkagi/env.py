from typing import Tuple, List, Optional
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import random
import math
from .consts import *
from .objs import Stone, Obstacle, SlingShot
from ..color import Color


class AlkkagiEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 60,
    }

    def __init__(self, n_stones: int, n_obstacles: int, render_mode: str):
        self.n_stones = n_stones
        self.n_obstacles = n_obstacles
        self.render_mode = render_mode

        self._space = pymunk.Space()
        self._space.damping = DAMPING

        self._stones: List[Stone] = []
        self._obstacles: List[Obstacle] = []

        self._turn = PLAYER_BLACK
        self._is_stone_in_motion = False

        self._screen = None
        self._game_surface = None
        self._status_surface = None
        self._clock = None

        self._slingshot = SlingShot()
        self._steps = 0

        self.width = GAME_WIDTH
        self.height = GAME_HEIGHT
        self.max_power = MAX_POWER

        self.observation_space = gym.spaces.Dict({
            'turn': gym.spaces.Discrete(2),
            'black': gym.spaces.Box(
                low=0,
                high=max(self.width, self.height),
                shape=(n_stones, 3),
                dtype=np.float32
            ),
            'white': gym.spaces.Box(
                low=0,
                high=max(self.width, self.height),
                shape=(n_stones, 3),
                dtype=np.float32
            ),
            'obstacles': gym.spaces.Box(
                low=0,
                high=max(self.width, self.height),
                shape=(n_stones, 4),
                dtype=np.float32
            )
        })

        self.action_space = gym.spaces.Dict({
            'turn': gym.spaces.Discrete(2),
            'index': gym.spaces.Box(0, n_stones, shape=(1,), dtype=np.uint8),
            'power': gym.spaces.Box(1, self.max_power, shape=(1,)),
            'angle': gym.spaces.Box(-180, 180, shape=(1,)),
        })

    @property
    def turn_(self) -> int:
        return self._turn

    @property
    def stone_in_motion_(self) -> bool:
        return self._is_stone_in_motion

    @property
    def latest_obs_(self):
        return self._generate_obs()

    @property
    def latest_info_(self):
        return self._generate_info()

    def set_slingshot_from_pos(
            self,
            pos_start: Tuple[float, float] | pymunk.Vec2d | None,
            pos_end: Tuple[float, float] | pymunk.Vec2d | None
    ):
        self._slingshot.pos_start = pos_start
        self._slingshot.pos_end = pos_end

    def set_slingshot_from_angle_power(
            self,
            pos_start: Tuple[float, float] | pymunk.Vec2d | None,
            angle: float,
            power: float
    ):
        sx, sy = pos_start
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        length = power / self.max_power * SlingShot.POWER_LINE_LENGTH
        ex, ey = sx - dx * length, sy - dy * length
        self.set_slingshot_from_pos((sx, sy), (ex, ey))

    def find_stone(self, pos: Tuple[float, float]) -> Stone:
        mx, my = pos

        closest_stone = None
        min_distance = float('inf')

        for stone in self._stones:
            if stone.player_ != self._turn:
                continue

            x, y = stone.position_
            dx, dy = x - mx, y - my
            distance = math.hypot(dx, dy)

            if distance <= Stone.RADIUS and distance < min_distance:
                closest_stone = stone
                min_distance = distance

        return closest_stone

    def reset(self, **kwargs):
        for stone in self._stones:
            if stone.active:
                self._space.remove(stone.body_, stone.shape_)

        for obstacle in self._obstacles:
            self._space.remove(obstacle.body_, obstacle.shape_)

        self._stones.clear()
        self._obstacles.clear()

        spacing = (self.height - 200) / (self.n_stones - 1)
        x, y = GRID_SIZE * 3, 100

        for i in range(self.n_stones):
            stone = Stone(x, y, PLAYER_BLACK, i)
            self._space.add(stone.body_, stone.shape_)
            self._stones.append(stone)
            y += spacing

        x, y = self.width - GRID_SIZE * 3, 100
        for i in range(self.n_stones):
            stone = Stone(x, y, PLAYER_WHITE, i)
            self._space.add(stone.body_, stone.shape_)
            self._stones.append(stone)
            y += spacing

        if self.n_obstacles > 0:
            spacing = (self.height - 200) / (self.n_obstacles - 1)
            x, y = self.width / 2, 100
            width, height = 10, 80
            for i in range(self.n_obstacles):
                obstacle = Obstacle(x, y, width, height)
                self._space.add(obstacle.body_, obstacle.shape_)
                self._obstacles.append(obstacle)
                y += spacing

        self._turn = PLAYER_BLACK
        self._steps = 0
        self.set_slingshot_from_pos(None, None)
        self.render()
        return self._generate_obs(), self._generate_info()

    def step(self, action):
        turn, index, power, angle = action['turn'], action['index'], action['power'], action['angle']

        if turn != self._turn:
            return self._generate_obs(), 0, False, False, self._generate_info()

        angle = np.clip(angle, -180, 180)
        power = np.clip(power, 1, self.max_power)
        stones = [stone for stone in self._stones if stone.player_ == self._turn and stone.active]
        selected_stone = stones[0]

        for stone in stones:
            if stone.index_ == index:
                selected_stone = stone
                break

        self.set_slingshot_from_angle_power(selected_stone.position_, angle, power)

        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        impulse = power * dx, power * dy

        selected_stone.apply_impulse(impulse)
        self._is_stone_in_motion = True

        while self._is_stone_in_motion:
            self._space.step(1 / self.metadata['render_fps'])

            stones_oob = [
                stone
                for stone in self._stones
                if stone.active and
                   (stone.position_.x < 0 or stone.position_.x > self.width
                   or stone.position_.y < 0 or stone.position_.y > self.height)
            ]

            for stone in stones_oob:
                self._space.remove(stone.body_, stone.shape_)
                stone.active = False

            for stone in self._stones:
                if stone.velocity_.length < VELOCITY_THRESHOLD:
                    stone.stop()

            self._is_stone_in_motion = any(
                stone.active and stone.velocity_.length >= VELOCITY_THRESHOLD for stone in self._stones
            )

            if self.render_mode == 'human':
                self.render()

        self._turn = PLAYER_BLACK if self._turn == PLAYER_WHITE else PLAYER_WHITE
        self.set_slingshot_from_pos(None, None)

        terminated = self._check_win_condition() is not None

        self._steps += 1

        self.render()
        return self._generate_obs(), 0, terminated, False, self._generate_info()

    def render(self,):
        screen_width, screen_height = self.width, self.height + STATUS_HEIGHT
        if self._screen is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self._screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption(self.spec.id)
            else:
                self._screen = pygame.Surface((screen_width, screen_height))

            self._game_surface = self._screen.subsurface((0, 0, self.width, self.height))
            self._status_surface = self._screen.subsurface((0, self.height, screen_width, STATUS_HEIGHT))

        if self._clock is None:
            self._clock = pygame.time.Clock()

        self._draw_go_board(self._game_surface)

        for obstacle in self._obstacles:
            obstacle.draw(self._game_surface)

        for stone in self._stones:
            stone.draw(self._game_surface)

        self._slingshot.draw(self._game_surface)

        if self._is_stone_in_motion:
            self._draw_status(
                text=f"({self._steps + 1}-th Action) Stones are moving.",
                text_color=Color.RED,
                fill=Color.RED
            )
        else:
            if self._turn == PLAYER_BLACK:
                self._draw_status(
                    text=f"({self._steps + 1}-th Action) Black player's turn",
                    text_color=Color.BLACK,
                    fill=Color.BLACK
                )
            else:
                self._draw_status(
                    text=f"({self._steps + 1}-th Action) White player's turn",
                    text_color=Color.BLACK,
                    fill=Color.WHITE,
                    stroke=Color.BLACK,
                )

        winner = self._check_win_condition()

        if winner is not None:
            winner = 'Black' if winner == PLAYER_BLACK else 'White'
            self._draw_status(
                text=f"Winner is {winner} player!",
                text_color=Color.BLUE,
                fill=Color.BLUE
            )

        if self.render_mode == 'human':
            pygame.event.pump()
            self._clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

    def get_frame(self):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._game_surface)), axes=(1, 0, 2)
        )

    def close(self):
        if self._screen:
            pygame.display.quit()

        pygame.quit()

    def _generate_info(self):
        return {
            'steps': self._steps,
        }

    def _generate_obs(self):
        black = np.zeros((self.n_stones, 3), dtype=np.float32)
        white = np.zeros((self.n_stones, 3), dtype=np.float32)

        for stone in self._stones:
            if stone.player_ == PLAYER_BLACK:
                black[stone.index_ ] = (*stone.position_, 1 if stone.active else 0)
            elif stone.player_ == PLAYER_WHITE:
                white[stone.index_] = (*stone.position_, 1 if stone.active else 0)

        obstacles = np.zeros((self.n_obstacles, 4), dtype=np.float32)
        for i, obstacle in enumerate(self._obstacles):
            obstacles[i] = (*obstacle.position_, *obstacle.size_)

        return {
            'turn': self._turn,
            'black': black,
            'white': white,
            'obstacles': obstacles
        }

    def _check_win_condition(self) -> Optional[int]:
        black = sum(1 for stone in self._stones if stone.player_ == PLAYER_BLACK and stone.active)
        white = sum(1 for stone in self._stones if stone.player_ == PLAYER_WHITE and stone.active)

        if black == 0:
           return PLAYER_WHITE
        elif white == 0:
           return PLAYER_BLACK
        else:
           return None

    def _draw_go_board(self, surface: pygame.Surface):
        surface.fill(Color.WOOD)

        for y in range(10, self.height, 20):
            start_x = random.randint(0, 20)
            end_x = self.width - random.randint(0, 20)
            pygame.draw.line(surface, Color.GRAIN, (start_x, y), (end_x, y), 1)

        for x in range(0, self.width, GRID_SIZE):
            pygame.draw.line(surface, Color.BLACK, (x, 0), (x, self.height), 1)

        for y in range(0, self.height, GRID_SIZE):
            pygame.draw.line(surface, Color.BLACK, (0, y), (self.width, y), 1)

    def _draw_status(
            self,
            text: str,
            text_color: Tuple[int, int, int],
            fill: Tuple[int, int, int],
            stroke: Tuple[int, int, int] = None
    ):
        self._status_surface.fill(Color.WHITE)

        rect = pygame.Rect(10, 15, 20, 20)
        pygame.draw.circle(self._status_surface, fill, rect.center, 10)

        if stroke:
            pygame.draw.circle(self._status_surface, stroke, rect.center, 10, 1)  # Black outline

        font = pygame.font.SysFont(None, 24)
        text = font.render(text, True, text_color)
        self._status_surface.blit(text, (rect.right + 10, rect.centery - 8))

