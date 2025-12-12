import os
from enum import IntEnum
from typing import Tuple, List
import numpy as np
import pygame
import pygame.freetype
import random
import gymnasium as gym


GAME_WIDTH = 600
GAME_HEIGHT = 750

STATUS_WIDTH = GAME_WIDTH
STATUS_HEIGHT = 50

SCREEN_WIDTH = GAME_WIDTH
SCREEN_HEIGHT = GAME_HEIGHT + STATUS_HEIGHT

UNIT_PIXEL_SIZE = 8
UNIT_MARGIN = 2
SPRITE_COLOR_KEY = (146, 144, 255)
SCALE = 3

PLAYER_WIDTH = UNIT_PIXEL_SIZE * 2 * SCALE
PLAYER_HEIGHT = UNIT_PIXEL_SIZE * 2 * SCALE
PLAYER_MAX_SPEED = 10
PLAYER_ACCELERATION = 0.25
PLAYER_DECELERATION = 0.10
PLAYER_ZERO_SPEED_THRESHOLD = 0.05
PLAYER_MIN_ANIM_INTERVAL = 0.05
PLAYER_MAX_ANIM_INTERVAL = 0.1

ENEMY_WIDTH = UNIT_PIXEL_SIZE * 2 * SCALE
ENEMY_HEIGHT = UNIT_PIXEL_SIZE * 2 * SCALE
ENEMY_MIN_ACCELERATION = 0.01
ENEMY_MAX_ACCELERATION = 0.20
ENEMY_ANIM_INTERVAL = 0.1

PATH_PLAYER_SPRITE = os.path.join(os.path.dirname(__file__), 'assets', 'players.png')
PATH_ENEMY_SPRITE = os.path.join(os.path.dirname(__file__), 'assets', 'enemies.png')


def _load_image(
        sprite: pygame.Surface,
        size: Tuple[int, int],
        offset: Tuple[int, int],
        flip: bool = False,
):

    img = sprite.subsurface(*offset, *size)
    if flip:
        img = pygame.transform.flip(img, True, False)
    return img


class Player:
    def __init__(
            self,
            sprite: pygame.Surface,
            x: int,
            y: int,
    ):
        self._x = x
        self._y = y
        self._velocity = 0
        self._idx = 0
        self._prev_velocity = 0
        self._anim_interval = PLAYER_MAX_ANIM_INTERVAL
        self._last_anim_time = 0
        self._duration = 0

        width, height = UNIT_PIXEL_SIZE * 2 * SCALE, UNIT_PIXEL_SIZE * 2 * SCALE

        offset_x = 0
        offset_y = UNIT_PIXEL_SIZE * SCALE

        self._img_standing = _load_image(sprite, (width, height), (offset_x, offset_y), False)

        offset_x += (width + UNIT_MARGIN * 2 * SCALE)

        self._img_walking_left, self._img_walking_right = [], []

        for _ in range(3):
            img_right = _load_image(sprite, (width, height),(offset_x, offset_y), False)
            img_left = _load_image(sprite, (width, height), (offset_x, offset_y), True)
            self._img_walking_right.append(img_right)
            self._img_walking_left.append(img_left)
            offset_x += width + UNIT_MARGIN * SCALE

        offset_x += width * 2 + UNIT_MARGIN * 5 * SCALE

        self._img_dead = _load_image(sprite, (width, height),(offset_x, offset_y), False)

        self._size = width, height
        self.active = True

    @property
    def position_(self):
        return self._x, self._y

    @property
    def size_(self):
        return self._size

    @property
    def velocity_(self):
        return self._velocity

    def update(self, action: 'Actions'):
        self._prev_velocity = self._velocity

        if action == Actions.LEFT:
            if self._velocity < 0:
                self._velocity -= PLAYER_ACCELERATION
            else:
                self._velocity = -PLAYER_ACCELERATION
            self._velocity = np.clip(self._velocity, -PLAYER_MAX_SPEED, PLAYER_MAX_SPEED)
        elif action == Actions.RIGHT:
            if self._velocity > 0:
                self._velocity += PLAYER_ACCELERATION
            else:
                self._velocity = PLAYER_ACCELERATION
            self._velocity += PLAYER_ACCELERATION
            self._velocity = np.clip(self._velocity, -PLAYER_MAX_SPEED, PLAYER_MAX_SPEED)
        else:
            if self._velocity > 0:
                self._velocity -= PLAYER_DECELERATION
                self._velocity = np.clip(self._velocity, 0, self._velocity)
            else:
                self._velocity += PLAYER_DECELERATION
                self._velocity = np.clip(self._velocity, self._velocity, 0)

        self._x += self._velocity
        self._x = np.clip(self._x, 0, GAME_WIDTH - self._size[0])

    def draw(self, surface: pygame.Surface, dt: float):
        self._duration += dt

        if not self.active:
            img = self._img_dead
        else:
            if self._prev_velocity * self._velocity < 0:
                self._idx = 0

            if abs(self._velocity) < PLAYER_ZERO_SPEED_THRESHOLD:
                img = self._img_standing
            else:
                img = self._img_walking_left[self._idx] if self._velocity < 0 else self._img_walking_right[self._idx]
                if self._duration - self._last_anim_time >= self._anim_interval:
                    self._idx = (self._idx + 1) % len(self._img_walking_left)
                    self._anim_interval = PLAYER_MAX_ANIM_INTERVAL * (
                            PLAYER_MIN_ANIM_INTERVAL / PLAYER_MAX_ANIM_INTERVAL
                    ) ** (abs(self._velocity) / PLAYER_MAX_SPEED)
                    self._last_anim_time = self._duration

        surface.blit(img, (self._x, self._y))


class Enemy:
    def __init__(
            self,
            sprite: pygame.Surface,
            x: int,
            y: int,
            acceleration: float,
            flip: bool = False
    ):
        self._x = x
        self._y = y
        self._velocity = 0
        self._acceleration = acceleration
        self._duration = 0
        self._last_anim_time = 0
        self._idx = 0

        self.active = True

        width, height = UNIT_PIXEL_SIZE * 2 * SCALE, UNIT_PIXEL_SIZE * 2 * SCALE

        offset_x = 0
        offset_y = UNIT_PIXEL_SIZE * 19  * SCALE + UNIT_MARGIN * 6 * SCALE

        self._img = []
        for _ in range(2):
            img = _load_image(sprite, (width, height), (offset_x, offset_y), flip)
            self._img.append(img)
            offset_x += width + UNIT_MARGIN * SCALE

        self._size = width, height

    @property
    def position_(self):
        return self._x, self._y

    @property
    def size_(self):
        return self._size

    @property
    def acceleration_(self):
        return self._acceleration

    @property
    def velocity_(self):
        return self._velocity

    def update(self):
        self._y += self._velocity
        self._velocity += self._acceleration

        self.active = self._y < GAME_HEIGHT

    def draw(self, surface: pygame.Surface, dt: float):
        self._duration += dt
        img = self._img[self._idx]

        if self._duration - self._last_anim_time >= ENEMY_ANIM_INTERVAL:
            self._idx = (self._idx + 1) % len(self._img)
            self._last_anim_time = self._duration

        surface.blit(img, (self._x, self._y))


class GameState(IntEnum):
    INIT = 0
    PLAYING = 1
    GAME_OVER = 2
    GAME_CLEARED = 3


class Actions(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2


class Color:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

class AvoidBlurpEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 60,
    }

    def __init__(
            self,
            game_duration: float,
            init_spawn_interval: float,
            min_spawn_interval: float,
            max_spawns: int,
            prob_spawn_on_player: float = 0.0,
            max_spawn_duration: float = None,
            frame_skip: int = 4,
            render_mode=None
    ):
        self.render_mode = render_mode
        self.game_duration = game_duration
        self.init_spawn_interval = init_spawn_interval
        self.min_spawn_interval = min_spawn_interval
        self.max_spawns = max_spawns
        self.max_spawn_duration = min(max_spawn_duration, game_duration) if max_spawn_duration else game_duration
        self.prob_spawn_on_player = prob_spawn_on_player
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self._time_elapsed = 0
        self._frame_count = 0
        self._last_spawn_frame = 0
        self._last_action = 0
        self._spawn_interval = init_spawn_interval
        self._game_state = GameState.INIT

        player_sprite = pygame.image.load(PATH_PLAYER_SPRITE)
        player_sprite.set_colorkey(SPRITE_COLOR_KEY)
        player_sprite = pygame.transform.scale(
            player_sprite,
            (player_sprite.get_width() * SCALE, player_sprite.get_height() * SCALE)
        )

        enemy_sprite = pygame.image.load(PATH_ENEMY_SPRITE)
        enemy_sprite.set_colorkey(SPRITE_COLOR_KEY)
        enemy_sprite = pygame.transform.scale(
            enemy_sprite,
            (enemy_sprite.get_width() * SCALE, enemy_sprite.get_height() * SCALE)
        )

        self._player_sprite = player_sprite
        self._enemy_sprite = enemy_sprite

        self._player: Player | None = None
        self._enemies: List[Enemy] = []

        self._screen = None
        self._game_surface = None
        self._status_surface = None
        self._clock = None

        self.observation_space = gym.spaces.Dict({
            "player": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,)),
            "enemies": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_spawns, 6)),
        })
        self.action_space = gym.spaces.Discrete(3)

    @property
    def time_elapsed_(self):
        return self._time_elapsed

    @staticmethod
    def _draw_status(
            surface: pygame.Surface,
            text: str,
            color: Tuple[int, int, int],
    ):
        font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        text = font.render(text, True, color)
        surface_width, surface_height = surface.get_size()
        text_width, text_height = text.get_size()
        surface.blit(text, (10, ((surface_height - text_height) // 2)))

    def render(self):
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption(self.spec.id)
            else:
                self._screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

            self._game_surface = self._screen.subsurface((0, 0, GAME_WIDTH, GAME_HEIGHT))
            self._status_surface = self._screen.subsurface((0, GAME_HEIGHT, STATUS_WIDTH, STATUS_HEIGHT))

        if self._clock is None:
            self._clock = pygame.time.Clock()

        dt = 1.0 / self.metadata['render_fps']

        self._game_surface.fill(SPRITE_COLOR_KEY)
        self._status_surface.fill(Color.WHITE)

        self._player.draw(self._game_surface, dt)

        for enemy in self._enemies:
            enemy.draw(self._game_surface, dt)

        if self._game_state == GameState.GAME_CLEARED:
            self._draw_status(self._status_surface, "Game Clear!", Color.BLUE)
        elif self._game_state == GameState.GAME_OVER:
            self._draw_status(self._status_surface, f"Game Over! You survived for {self._time_elapsed:.2f} seconds", Color.RED)
        else:
            self._draw_status(self._status_surface, f"Time elapsed: {self._time_elapsed:.2f} seconds", Color.BLACK)

        if self.render_mode == "human":
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

    def reset(self, **kwargs):
        super().reset(**kwargs)

        self._player = Player(
            self._player_sprite, SCREEN_WIDTH // 2, GAME_HEIGHT - PLAYER_HEIGHT
        )
        self._enemies.clear()
        self._time_elapsed = 0
        self._frame_count = 0
        self._last_spawn_frame = 0
        self._last_action = 0
        self._spawn_interval = self.init_spawn_interval
        self._game_state = GameState.PLAYING

        self.render()

        return self._generate_obs(), dict()

    def _generate_obs(self):
        player = np.array(
            (*self._player.position_, *self._player.size_, self._player.velocity_),
            dtype=np.float32
        )
        enemies = np.zeros(shape=(self.max_spawns, 6), dtype=np.float32)

        for i, enemy in enumerate(self._enemies):
            enemies[i] = (*enemy.position_, *enemy.size_, enemy.velocity_, enemy.acceleration_)

        return {
            "player": player,
            "enemies": enemies,
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}

        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self._step_internal(action)
            if terminated or truncated:
                return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info

    def _step_internal(self, action):
        fps = self.metadata["render_fps"]
        dt = 1.0 / fps

        self._frame_count += 1
        self._last_action = action
        self._player.update(action)
        self._time_elapsed += dt

        if self._time_elapsed >= self.game_duration:
            self._game_state = GameState.GAME_CLEARED
            self.render()
            return self._generate_obs(), 0.0, True, False, dict()

        frames_per_spawn = int(self._spawn_interval * fps)
        if self._frame_count - self._last_spawn_frame >= frames_per_spawn:
            if random.uniform(0.0, 1.0) < self.prob_spawn_on_player:
                x = self._player.position_[0]
            else:
                x = random.randint(0, GAME_WIDTH - ENEMY_WIDTH)

            if len(self._enemies) < self.max_spawns:
                accel = random.uniform(ENEMY_MIN_ACCELERATION, ENEMY_MAX_ACCELERATION)
                flip = random.choice([True, False])
                enemy = Enemy(self._enemy_sprite, x, 0, accel, flip)
                self._enemies.append(enemy)

            self._last_spawn_frame = self._frame_count
            self._spawn_interval = self.init_spawn_interval * (
                    self.min_spawn_interval / self.init_spawn_interval
            ) ** (self._time_elapsed / self.max_spawn_duration)

        for enemy in self._enemies:
            enemy.update()
            ew, eh = enemy.size_
            ex, ey = enemy.position_
            pw, ph = self._player.size_
            px, py = self._player.position_

            if ey < GAME_HEIGHT and px < ex + ew and px + pw > ex and py < ey + eh and self._player.active:
                self._player.active = False
                self._game_state = GameState.GAME_OVER

        enemies_oob = [enemy for enemy in self._enemies if not enemy.active]

        for enemy in enemies_oob:
            self._enemies.remove(enemy)

        self.render()

        if self._game_state == GameState.PLAYING:
            return self._generate_obs(), 0.0, False, False, {"time_elapsed": self._time_elapsed}
        else:
            return self._generate_obs(), 0.0, False, True, {"time_elapsed": self._time_elapsed}

    def close(self):
        if self._screen is not None:
            pygame.display.quit()

        pygame.quit()
