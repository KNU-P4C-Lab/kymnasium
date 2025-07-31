from enum import IntEnum
from typing import Tuple, List, Optional
import numpy as np
import pygame
import pymunk
import random
import math
import gymnasium as gym
from pymunk import Vec2d


GAME_WIDTH = 600
GAME_HEIGHT = 600
STATUS_HEIGHT = 50
SCREEN_WIDTH = GAME_WIDTH
SCREEN_HEIGHT = GAME_HEIGHT + STATUS_HEIGHT

GRID_SIZE = 600 // 20

STONE_RADIUS = 15
STONE_MASS = 10
STONE_ELASTICITY = 0.85

OBSTACLE_ELASTICITY = 0.25

MAX_POWER = 2500
DAMPING = 0.6  # Velocity damping factor (lower value = more damping)
VELOCITY_THRESHOLD = 5.0  # Threshold below which stones are considered "at rest"

POWER_LINE_LENGTH = 100


class Player(IntEnum):
    black = 0
    white = 1


class Color:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BRASS = (181, 166, 66)
    WOOD = (219, 169, 97)
    GRAIN = (200, 150, 80)
    GREY = (128, 128, 128)
    LIGHT_GREY = (211, 211, 211)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)


class Stone:
    def __init__(self, x: float, y: float, player: Player, index: int):
        moment = pymunk.moment_for_circle(STONE_MASS, 0, STONE_RADIUS)

        self._body = pymunk.Body(STONE_MASS, moment)
        self._body.position = (x, y)

        self._shape = pymunk.Circle(self._body, STONE_RADIUS)
        self._shape.elasticity = STONE_ELASTICITY

        self._player = player
        self._index = index

        self.active = True

    def draw(self, surface: pygame.Surface):
        if not self.active:
            return

        font = pygame.font.SysFont(None, 24)
        if self._player == Player.black:
            pygame.draw.circle(surface, Color.BLACK, self._body.position, STONE_RADIUS)  # Black
            text_surface = font.render(str(self._index), True, Color.WHITE)
        else:
            pygame.draw.circle(surface, Color.WHITE, self._body.position, STONE_RADIUS)  # White
            pygame.draw.circle(surface, Color.BLACK, self._body.position, STONE_RADIUS, 1)  # Black outline
            text_surface = font.render(str(self._index), True, Color.BLACK)

        text_rect = text_surface.get_rect(center=self.position_)
        surface.blit(text_surface, text_rect)

    def apply_impulse(self, impulse: Tuple[float, float] = (0, 0)):
        self._body.apply_impulse_at_local_point(impulse, (0, 0))

    def stop(self):
        self._body.velocity = (0, 0)

    @property
    def position_(self):
        return self._body.position

    @property
    def index_(self):
        return self._index

    @property
    def player_(self):
        return self._player

    @property
    def shape_(self):
        return self._shape

    @property
    def body_(self):
        return self._body

    @property
    def velocity_(self):
        return self._body.velocity


class Obstacle:
    def __init__(self, x: float, y: float, width: float, height: float):
        self._width = width
        self._height = height

        self._body = pymunk.Body(body_type=pymunk.Body.STATIC)  # Static body (doesn't move)
        self._body.position = (x, y)

        self._shape = pymunk.Poly.create_box(self._body, (width, height))
        self._shape.elasticity = OBSTACLE_ELASTICITY  # Less bouncy for wooden board feel

    def draw(self, surface: pygame.Surface):
        x, y = self._body.position
        rect = pygame.Rect(x - self._width / 2, y - self._height / 2, self._width, self._height)
        pygame.draw.rect(surface, Color.BRASS, rect)
        pygame.draw.rect(surface, Color.BLACK, rect, 1)  # Black outline

    @property
    def size_(self):
        return self._width, self._height

    @property
    def position_(self):
        return self._body.position

    @property
    def shape_(self):
        return self._shape

    @property
    def body_(self):
        return self._body


class SlingShot:
    def __init__(self,
                 pos_start: Tuple[float, float] | Vec2d | None = None,
                 pos_end: Tuple[float, float] | Vec2d | None = None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.wait = False

    def draw(self, surface: pygame.Surface):
        if self.pos_start is None or self.pos_end is None:
            return

        sx, sy = self.pos_start
        ex, ey = self.pos_end
        dx, dy = sx - ex, sy - ey
        distance = math.hypot(dx, dy)

        if distance <= 0:
            return

        nx, ny = dx / distance, dy / distance

        len_arrow, len_dash = 75, 5
        n_dashes = int(len_arrow / (2 * len_dash))

        ex, ey = sx + nx * len_arrow, sy + ny * len_arrow

        for i in range(n_dashes):
            start_ratio = i * (2 * len_dash) / len_arrow
            end_ratio = min(start_ratio + len_dash / len_arrow, 1)
            dash_start = (
                sx + (ex - sx) * start_ratio,
                sy + (ey - sy) * start_ratio
            )
            dash_end = (
                sx + (ex - sx) * end_ratio,
                sy + (ey - sy) * end_ratio
            )

            pygame.draw.line(surface, Color.LIGHT_GREY, dash_start, dash_end, 2)

        arrow_size = 10
        arrow_angle = 30  # degrees

        # Calculate arrow head points
        angle = pygame.math.Vector2(0, 0).angle_to(pygame.math.Vector2(nx, ny))
        l_angle, r_angle = math.radians(angle + arrow_angle), math.radians(angle - arrow_angle)

        arrow_left = (
            ex - arrow_size * math.cos(l_angle),
            ey - arrow_size * math.sin(l_angle)
        )
        arrow_right = (
            ex - arrow_size * math.cos(r_angle),
            ey - arrow_size * math.sin(r_angle)
        )

        pygame.draw.line(surface, Color.LIGHT_GREY, (ex, ey), arrow_left, 2)
        pygame.draw.line(surface, Color.LIGHT_GREY, (ex, ey), arrow_right, 2)

        len_power = min(distance, POWER_LINE_LENGTH)
        power_end = (
            sx - nx * len_power,
            sy - ny * len_power
        )

        pygame.draw.line(surface, Color.LIGHT_GREY, (sx, sy), power_end, 3)


class AlkkagiEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 60
    }

    def __init__(self, n_stones: int, n_obstacles: int, render_mode: str):
        self.n_stones = n_stones
        self.n_obstacles = n_obstacles
        self.render_mode = render_mode

        self._space = pymunk.Space()
        self._space.damping = DAMPING

        self._stones: List[Stone] = []
        self._obstacles: List[Obstacle] = []

        self._turn: Player = Player.black
        self._is_stone_in_motion = False

        self._screen = None
        self._game_surface = None
        self._status_surface = None
        self._clock = None

        self._slingshot = SlingShot()
        self._steps = 0

        self.observation_space = gym.spaces.Dict({
            'turn': gym.spaces.Discrete(2),
            'black': gym.spaces.Box(
                low=0,
                high=max(GAME_WIDTH, GAME_HEIGHT),
                shape=(n_stones, 3),
                dtype=np.float32
            ),
            'white': gym.spaces.Box(
                low=0,
                high=max(GAME_WIDTH, GAME_HEIGHT),
                shape=(n_stones, 3),
                dtype=np.float32
            ),
            'obstacles': gym.spaces.Box(
                low=0,
                high=max(GAME_WIDTH, GAME_HEIGHT),
                shape=(n_stones, 4),
                dtype=np.float32
            )
        })

        self.action_space = gym.spaces.Dict({
            'turn': gym.spaces.Discrete(2),
            'index': gym.spaces.Box(0, n_stones, shape=(1,), dtype=np.uint8),
            'power': gym.spaces.Box(1, MAX_POWER, shape=(1,)),
            'angle': gym.spaces.Box(-180, 180, shape=(1,)),
        })

    @property
    def turn_(self) -> Player:
        return self._turn

    @property
    def stone_in_motion_(self) -> bool:
        return self._is_stone_in_motion

    def set_slingshot_from_pos(
            self,
            pos_start: Tuple[float, float] | Vec2d | None,
            pos_end: Tuple[float, float] | Vec2d | None
    ):
        self._slingshot.pos_start = pos_start
        self._slingshot.pos_end = pos_end

    def set_slingshot_from_angle_power(
            self,
            pos_start: Tuple[float, float] | Vec2d | None,
            angle: float,
            power: float
    ):
        sx, sy = pos_start
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        length = power / MAX_POWER * POWER_LINE_LENGTH
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

            if distance <= STONE_RADIUS and distance < min_distance:
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

        spacing = (GAME_HEIGHT - 200) / (self.n_stones - 1)
        x, y = GRID_SIZE * 3, 100

        for i in range(self.n_stones):
            stone = Stone(x, y, Player.black, i)
            self._space.add(stone.body_, stone.shape_)
            self._stones.append(stone)
            y += spacing

        x, y = GAME_WIDTH - GRID_SIZE * 3, 100
        for i in range(self.n_stones):
            stone = Stone(x, y, Player.white, i)
            self._space.add(stone.body_, stone.shape_)
            self._stones.append(stone)
            y += spacing

        if self.n_obstacles > 0:
            spacing = (GAME_HEIGHT - 200) / (self.n_obstacles - 1)
            x, y = GAME_WIDTH / 2, 100
            width, height = 10, 80
            for i in range(self.n_obstacles):
                obstacle = Obstacle(x, y, width, height)
                self._space.add(obstacle.body_, obstacle.shape_)
                self._obstacles.append(obstacle)
                y += spacing

        self._turn = Player.black
        self._steps = 0
        self.set_slingshot_from_pos(None, None)
        self.render()
        return self._generate_obs(), self._generate_info()

    def step(self, action):
        turn, index, power, angle = action['turn'], action['index'], action['power'], action['angle']

        if turn != self._turn:
            return self._generate_obs(), 0, False, False, self._generate_info()

        angle = np.clip(angle, -180, 180)
        power = np.clip(power, 1, MAX_POWER)
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
                   (stone.position_.x < 0 or stone.position_.x > GAME_WIDTH
                   or stone.position_.y < 0 or stone.position_.y > GAME_HEIGHT)
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

        self._turn = Player.black if self._turn == Player.white else Player.white
        self.set_slingshot_from_pos(None, None)

        terminated = self._check_win_condition() is not None

        self._steps += 1

        self.render()
        return self._generate_obs(), 0, terminated, False, self._generate_info()

    def render(self,):
        if self._screen is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption(self.spec.id)
            else:
                self._screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

            self._game_surface = self._screen.subsurface((0, 0, GAME_WIDTH, GAME_HEIGHT))
            self._status_surface = self._screen.subsurface((0, GAME_HEIGHT, SCREEN_WIDTH, STATUS_HEIGHT))

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
                self._status_surface,
                text="Stones are moving.",
                text_col=Color.RED,
                fill=Color.RED
            )
        else:
            if self._turn == Player.black:
                self._draw_status(
                    self._status_surface,
                    text="Black player's turn",
                    text_col=Color.BLACK,
                    fill=Color.BLACK
                )
            else:
                self._draw_status(
                    self._status_surface,
                    text="White player's turn",
                    text_col=Color.BLACK,
                    fill=Color.WHITE,
                    stroke=Color.BLACK,
                )

        winner = self._check_win_condition()

        if winner is not None:
            winner = 'Black' if winner == Player.black else 'White'
            self._draw_status(
                self._status_surface,
                text=f"Winner is {winner} player!",
                text_col=Color.BLUE,
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
            if stone.player_ == Player.black:
                black[stone.index_ ] = (*stone.position_, 1 if stone.active else 0)
            elif stone.player_ == Player.white:
                white[stone.index_] = (*stone.position_, 1 if stone.active else 0)

        obstacles = np.zeros((self.n_obstacles, 4), dtype=np.float32)
        for i, obstacle in enumerate(self._obstacles):
            obstacles[i] = (*obstacle.position_, *obstacle.size_)

        return {
            'turn': self._turn.value,
            'black': black,
            'white': white,
            'obstacles': obstacles
        }

    def _check_win_condition(self) -> Optional[Player]:
        black = sum(1 for stone in self._stones if stone.player_ == Player.black and stone.active)
        white = sum(1 for stone in self._stones if stone.player_ == Player.white and stone.active)

        if black == 0:
           return Player.white
        elif white == 0:
           return Player.black
        else:
           return None

    @staticmethod
    def _draw_go_board(surface: pygame.Surface):
        surface.fill(Color.WOOD)

        for y in range(10, GAME_HEIGHT, 20):
            start_x = random.randint(0, 20)
            end_x = GAME_WIDTH - random.randint(0, 20)
            pygame.draw.line(surface, Color.GRAIN, (start_x, y), (end_x, y), 1)

        for x in range(0, GAME_WIDTH, GRID_SIZE):
            pygame.draw.line(surface, Color.BLACK, (x, 0), (x, GAME_HEIGHT), 1)

        for y in range(0, GAME_HEIGHT, GRID_SIZE):
            pygame.draw.line(surface, Color.BLACK, (0, y), (GAME_WIDTH, y), 1)

    @staticmethod
    def _draw_status(surface: pygame.Surface,
                     text: str,
                     text_col: Tuple[int, int, int],
                     fill: Tuple[int, int, int],
                     stroke: Tuple[int, int, int] = None
                     ):
        surface.fill(Color.WHITE)

        rect = pygame.Rect(10, 15, 20, 20)
        pygame.draw.circle(surface, fill, rect.center, 10)

        if stroke:
            pygame.draw.circle(surface, stroke, rect.center, 10, 1)  # Black outline

        font = pygame.font.SysFont(None, 24)
        text = font.render(text, True, text_col)
        surface.blit(text, (rect.right + 10, rect.centery - 8))

