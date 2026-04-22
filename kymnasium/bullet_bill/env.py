import random
from typing import Tuple, List, Literal
import numpy as np
import pygame
import pygame.freetype
import gymnasium as gym
from .objs import Mario, BulletBill, Star, WorldObjects
from .consts import *
from ..color import Color
from ..sprite import load_tile_map_data, Group


class BulletBillEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': FPS
    }

    def __init__(
            self,
            game_duration: float,
            init_spawn_interval: float,
            min_spawn_interval: float,
            max_spawns: int,
            stage: int = 1,
            continuous_action: bool = False,
            max_spawn_duration: float = None,
            frame_skip: int = 4,
            render_mode=None
    ):
        self.render_mode = render_mode
        self.stage = stage
        self.continuous_action = continuous_action
        self.max_spawn_duration = min(max_spawn_duration, game_duration) if max_spawn_duration else game_duration
        self.frame_skip = frame_skip
        self.game_duration = game_duration
        self.init_spawn_interval = init_spawn_interval
        self.min_spawn_interval = min_spawn_interval
        self.max_spawns = max_spawns

        self._time_elapsed = 0
        self._timer = 0
        self._spawn_interval = init_spawn_interval
        self._game_state: Literal['init', 'playing', 'game_over', 'cleared'] = 'init'

        sprite = pygame.image.load(PATH_SPRITE)
        sprite.set_colorkey(SPRITE_COLOR_KEY)
        sprite = pygame.transform.scale(
            sprite,
            (sprite.get_width() * SCALE, sprite.get_height() * SCALE)
        )
        self._sprite = sprite

        self._screen = None
        self._game_surface = None
        self._status_surface = None
        self._clock = pygame.time.Clock()

        self._background_map = load_tile_map_data(
            os.path.join(os.path.dirname(__file__), 'assets', f'stage-{self.stage}{FILE_BACKGROUND}'),
            TILE_SIZE,
            self._sprite,
            SCALE
        )

        self._object_map = load_tile_map_data(
            os.path.join(os.path.dirname(__file__), 'assets', f'stage-{self.stage}{FILE_OBJECT}'),
            TILE_SIZE,
            self._sprite,
            SCALE
        )

        self.width, self.height = self._background_map.width, self._background_map.height

        self._character_sprites = Group()
        self._background_sprites = Group()
        self._static_sprites = Group()

        self._mario: Mario | None = None
        self._bullet_bills: List[BulletBill] = []
        self._stars: List[Star] = []
        self._platforms: List[WorldObjects] = []

        self._n_star = sum([tile.identifier == Ids.star for tile in self._object_map.tiles])

        self.observation_space = gym.spaces.Dict({
            "mario": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,)),
            "bullet_bills": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_spawns, 5)),
            "stars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._n_star, 4)),
        })
        if self.continuous_action:
            self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        else:
            self.action_space = gym.spaces.Discrete(4)

    def render(self):
        fps = self.metadata["render_fps"]
        dt = 1.0 / fps

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((self.width, self.height + STATUS_HEIGHT))
                pygame.display.set_caption(self.spec.id)
            else:
                self._screen = pygame.Surface((self.width, self.width + STATUS_HEIGHT), pygame.SRCALPHA)
            self._game_surface = self._screen.subsurface((0, 0, self.width, self.height))
            self._status_surface = self._screen.subsurface((0, self.height, self.width, STATUS_HEIGHT))

        self._game_surface.fill(Color.SKY_BLUE)

        self._character_sprites.update(dt)

        self._background_sprites.draw(self._game_surface)
        self._static_sprites.draw(self._game_surface)
        self._character_sprites.draw(self._game_surface)

        if self._game_state == 'cleared':
            self._draw_status("Game Clear!", Color.BLUE)
        elif self._game_state == 'game_over':
            self._draw_status(f"Game Over! You survived for {self._time_elapsed:.2f} seconds", Color.RED)
        else:
            self._draw_status(f"Time elapsed: {self._time_elapsed:.2f} seconds", Color.BLACK)

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

        self._character_sprites.empty()
        self._background_sprites.empty()
        self._static_sprites.empty()

        self._mario = None
        self._bullet_bills.clear()
        self._stars.clear()

        self._time_elapsed = 0
        self._timer = 0
        self._spawn_interval = self.init_spawn_interval

        self._build_stage()
        self._game_state = 'playing'

        self.render()

        return self._generate_obs(), dict()

    def step(self, action):
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}

        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self._step_internal(action)
            if terminated or truncated:
                break
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._screen is not None:
            pygame.display.quit()

        pygame.quit()

    def _build_stage(self):
        for tile in self._background_map.tiles:
            background = WorldObjects(tile.sprite,  (tile.x, tile.y))
            self._background_sprites.add(background)

        for tile in self._object_map.tiles:
            if tile.identifier == Ids.mario:
                self._mario = Mario(
                    sprite=self._sprite,
                    position=(tile.x, tile.y)
                )
                self._character_sprites.add(self._mario)
            elif tile.identifier == Ids.star:
                star = Star(
                    self._sprite, (tile.x, tile.y)
                )
                self._character_sprites.add(star)
                self._stars.append(star)
            else:
                platform = WorldObjects(tile.sprite, (tile.x, tile.y))
                self._static_sprites.add(platform)
                self._platforms.append(platform)

    def _spawn_bullet_bill(self):
        if len(self._bullet_bills) >= self.max_spawns:
            return

        target_y = random.uniform(-GAME_TILE_SIZE * 3, GAME_TILE_SIZE * 3) + self._mario.bb_[1]
        target_y = np.clip(target_y, 0, self._mario.bb_[1])
        is_left = random.choice([True, False])
        x = -GAME_TILE_SIZE if is_left else self.width
        speed = random.uniform(BULLET_BILL_MIN_SPEED, BULLET_BILL_MAX_SPEED)
        bullet_bill = BulletBill(
            sprite=self._sprite,
            position=(x, target_y),
            velocity=speed if is_left else -speed,
            flip=is_left,
        )
        self._bullet_bills.append(bullet_bill)
        self._character_sprites.add(bullet_bill)

    def _generate_obs(self):
        mario = np.array((
            *self._mario.bb_, *self._mario.velocity_,
            1 if self._mario.invincible_ else 0, 1 if self._mario.jumping_ else 0
        ), dtype=np.float32)

        bullets = np.zeros(shape=(self.max_spawns, 5), dtype=np.float32)
        bullets_active = [bullet for bullet in self._bullet_bills if bullet.state_ == 'active']

        for i, bullet in enumerate(bullets_active):
            bullets[i] = (*bullet.bb_, bullet.velocity_)

        stars = np.zeros(shape=(self._n_star, 4), dtype=np.float32)
        stars_active = [star for star in self._stars if star.active_]

        for i, star in enumerate(stars_active):
            stars[i] = (*star.bb_, )

        return {
            "mario": mario,
            "bullet_bills": bullets,
            "stars": stars,
        }

    def _draw_status(
            self,
            text: str,
            color: Tuple[int, int, int],
    ):
        font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        text = font.render(text, True, color)
        surface_width, surface_height = self._status_surface.get_size()
        text_width, text_height = text.get_size()
        self._status_surface.fill(Color.WHITE)
        self._status_surface.blit(text, (10, ((surface_height - text_height) // 2)))

    def _step_internal(self, action):
        fps = self.metadata["render_fps"]
        dt = 1.0 / fps

        self._timer += dt
        self._time_elapsed += dt

        if self._time_elapsed >= self.game_duration:
            self._game_state = 'cleared'
            self.render()
            return self._generate_obs(), 0.0, True, False, dict()

        if not self.continuous_action:
            walk_force, jump_force = 0.0, 0.0
            if action == Actions.left:
                walk_force = -MARIO_DEFAULT_MOVING_FORCE
            elif action == Actions.right:
                walk_force = MARIO_DEFAULT_MOVING_FORCE
            elif action == Actions.jump:
                jump_force = -MARIO_JUMPING_FORCE

            action = (walk_force, jump_force)

        self._mario.apply_force(action)

        if self._timer >= self._spawn_interval:
            self._timer = 0
            self._spawn_bullet_bill()

        self._spawn_interval = self.init_spawn_interval * (
                self.min_spawn_interval / self.init_spawn_interval
        ) ** (self._time_elapsed / self.max_spawn_duration)

        self._character_sprites.frame_tick(dt=dt, screen_size=(self.width, self.height), platforms=self._platforms)

        self._mario.check_star_collision(self._stars)
        self._mario.check_bullet_collision(self._bullet_bills)

        bullets_oob = [bullet for bullet in self._bullet_bills if bullet.state_ == 'oob']
        for bullet in bullets_oob:
            bullet.kill()
            self._bullet_bills.remove(bullet)

        stars_inactive = [star for star in self._stars if not star.active_]
        for star in stars_inactive:
            star.kill()
            self._stars.remove(star)

        if not self._mario.active_:
            self._game_state = 'game_over'

        self.render()

        if self._game_state == 'playing':
            return self._generate_obs(), 0.0, False, False, {"time_elapsed": self._time_elapsed}
        else:
            return self._generate_obs(), 0.0, False, True, {"time_elapsed": self._time_elapsed}