from typing import Tuple, List, Literal
import numpy as np
import pygame
import gymnasium as gym
from .objs import Mario, BulletBill, Star, WorldObjects
from .consts import *
from ...common.color import Color
from ...common.sprite import load_tile_map_data, Group
from ...common.types import RenderMode, ObsType


class BulletBillEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'none'],
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
            max_spawn_duration: float = 0.0,
            frame_skip: int = 4,
            render_mode: RenderMode = 'human',
            obs_type: ObsType = 'default',
            seed: int | None = None
    ):
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.should_render = self.render_mode != 'none' or self.obs_type == 'image'
        self.stage = stage
        self.continuous_action = continuous_action
        self.max_spawn_duration = min(max_spawn_duration, game_duration) if max_spawn_duration else game_duration
        self.frame_skip = frame_skip
        self.game_duration = game_duration
        self.init_spawn_interval = init_spawn_interval
        self.min_spawn_interval = min_spawn_interval
        self.max_spawns = max_spawns

        self._seed = seed
        self._random = np.random.default_rng(self._seed)

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
        self._mario_images = Mario.load_images(sprite)
        self._bullet_bill_images = BulletBill.load_images(sprite)
        self._star_images = Star.load_images(sprite)

        self._screen = None
        self._game_surface = None
        self._status_surface = None
        self._font = None
        self._clock = pygame.time.Clock()

        self._background_map = load_tile_map_data(
            str(ASSET_DIR / f'stage-{self.stage}{FILE_BACKGROUND}'),
            TILE_SIZE,
            self._sprite,
            SCALE
        )

        self._object_map = load_tile_map_data(
            str(ASSET_DIR / f'stage-{self.stage}{FILE_OBJECT}'),
            TILE_SIZE,
            self._sprite,
            SCALE
        )

        self.width, self.height = self._background_map.width, self._background_map.height
        self._screen_size = (self.width, self.height)

        self._character_sprites = Group()
        self._background_sprites = Group()
        self._static_sprites = Group()

        self._mario: Mario | None = None
        self._bullet_bills: List[BulletBill] = []
        self._stars: List[Star] = []
        self._platforms: List[WorldObjects] = []
        self._platform_grid: dict[tuple[int, int], WorldObjects] = {}
        self._nearby_platform_cache_key: tuple[int, int, int, int] | None = None
        self._nearby_platform_cache: List[WorldObjects] = []
        self._mario_spawn_position: Tuple[int, int] | None = None
        self._star_spawn_positions: List[Tuple[int, int]] = []

        self._n_star = sum([tile.identifier == SpriteId.star for tile in self._object_map.tiles])
        self._build_static_stage()
        self._reset_dynamic_stage()

        if self.obs_type == 'default':
            self.observation_space = gym.spaces.Dict({
                "mario": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,)),
                "bullet_bills": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_spawns, 5)),
                "stars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._n_star, 4)),
            })
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

        if self.continuous_action:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Discrete(len(Action))

    def render(self):
        if not self.should_render:
            return None

        dt = 1.0 / FPS

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((self.width, self.height + STATUS_HEIGHT))
                pygame.display.set_caption(self.spec.id)
            else:
                self._screen = pygame.Surface((self.width, self.height + STATUS_HEIGHT), pygame.SRCALPHA)
            self._game_surface = self._screen.subsurface((0, 0, self.width, self.height))
            self._status_surface = self._screen.subsurface((0, self.height, self.width, STATUS_HEIGHT))

        self._character_sprites.update(dt)

        assert self._game_surface is not None, "Game surface must be initialized before rendering"

        self._game_surface.fill(Color.SKY_BLUE)

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
            self._clock.tick(FPS)
            pygame.display.flip()
            return None
        return None

    def get_frame(self):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._game_surface)), axes=(1, 0, 2)
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)

        seed = kwargs.get('seed')
        if seed is not None:
            self._random = np.random.default_rng(seed)
        else:
            self._random = np.random.default_rng(self._seed)

        self._character_sprites.empty()

        self._mario = None
        self._bullet_bills.clear()
        self._stars.clear()

        self._time_elapsed = 0
        self._timer = 0
        self._spawn_interval = self.init_spawn_interval

        self._reset_dynamic_stage()
        self._game_state = 'playing'

        if self.should_render:
            self.render()

        return self._generate_obs(), self._generate_info()

    def step(self, action):
        action_force = self._action_to_force(action)

        for _ in range(self.frame_skip):
            if self._step_internal(action_force):
                break

        terminated = self._game_state == 'cleared'
        truncated = self._game_state == 'game_over'

        return self._generate_obs(), 0.0, terminated, truncated, self._generate_info()

    def close(self):
        if self._screen is not None:
            pygame.display.quit()

        pygame.quit()

    def _action_to_force(self, action) -> Tuple[float, float]:
        if self.continuous_action:
            return float(action[0] * MARIO_MAX_MOVING_FORCE), -float(action[1] * MARIO_JUMPING_FORCE)

        if action == Action.left:
            return -MARIO_DEFAULT_MOVING_FORCE, 0.0
        if action == Action.right:
            return MARIO_DEFAULT_MOVING_FORCE, 0.0
        if action == Action.jump:
            return 0.0, -MARIO_JUMPING_FORCE
        return 0.0, 0.0

    def _build_static_stage(self):
        for tile in self._background_map.tiles:
            background = WorldObjects(tile.sprite,  (tile.x, tile.y))
            self._background_sprites.add(background)

        for tile in self._object_map.tiles:
            if tile.identifier == SpriteId.mario:
                self._mario_spawn_position = (tile.x, tile.y)
            elif tile.identifier == SpriteId.star:
                self._star_spawn_positions.append((tile.x, tile.y))
            else:
                platform = WorldObjects(tile.sprite, (tile.x, tile.y))
                self._static_sprites.add(platform)
                self._platforms.append(platform)
                self._platform_grid[(tile.grid_x, tile.grid_y)] = platform

        if self._mario_spawn_position is None:
            raise ValueError(f"stage {self.stage} does not define a Mario spawn tile")

    def _nearby_platforms(self) -> List[WorldObjects]:
        assert self._mario is not None

        rect = self._mario.rect
        margin_tiles = 2
        min_x = max(0, rect.left // GAME_TILE_SIZE - margin_tiles)
        max_x = min((self.width - 1) // GAME_TILE_SIZE, rect.right // GAME_TILE_SIZE + margin_tiles)
        min_y = max(0, rect.top // GAME_TILE_SIZE - margin_tiles)
        max_y = min((self.height - 1) // GAME_TILE_SIZE, rect.bottom // GAME_TILE_SIZE + margin_tiles)
        cache_key = (min_x, max_x, min_y, max_y)

        if cache_key == self._nearby_platform_cache_key:
            return self._nearby_platform_cache

        platforms = []
        for grid_x in range(min_x, max_x + 1):
            for grid_y in range(min_y, max_y + 1):
                platform = self._platform_grid.get((grid_x, grid_y))
                if platform is not None:
                    platforms.append(platform)
        self._nearby_platform_cache_key = cache_key
        self._nearby_platform_cache = platforms
        return platforms

    def _reset_dynamic_stage(self):
        assert self._mario_spawn_position is not None

        self._nearby_platform_cache_key = None
        self._mario = Mario(
            images=self._mario_images,
            position=self._mario_spawn_position,
        )
        self._character_sprites.add(self._mario)

        for position in self._star_spawn_positions:
            star = Star(self._star_images, position)
            self._character_sprites.add(star)
            self._stars.append(star)

    def _spawn_bullet_bill(self):
        assert self._mario is not None

        if len(self._bullet_bills) >= self.max_spawns:
            return

        mario_top = self._mario.rect.top
        target_y = self._random.uniform(-GAME_TILE_SIZE * 3, GAME_TILE_SIZE * 3) + mario_top
        target_y = max(0, min(mario_top, target_y))
        is_left = self._random.random() < 0.5
        x = -GAME_TILE_SIZE if is_left else self.width
        speed = self._random.uniform(BULLET_BILL_MIN_SPEED, BULLET_BILL_MAX_SPEED)
        bullet_bill = BulletBill(
            images=self._bullet_bill_images,
            position=(x, target_y),
            velocity=speed if is_left else -speed,
            flip=is_left,
        )
        self._bullet_bills.append(bullet_bill)
        self._character_sprites.add(bullet_bill)

    def _generate_obs(self):
        assert self._mario is not None, "Mario must be initialized before generating observations"

        if self.obs_type == 'image':
            return self.get_frame()
        elif self.obs_type == 'default':
            mario = np.empty(shape=(8,), dtype=np.float32)
            mario_rect = self._mario.rect
            mario_velocity = self._mario.velocity_
            mario[0] = mario_rect.left
            mario[1] = mario_rect.top
            mario[2] = mario_rect.right
            mario[3] = mario_rect.bottom
            mario[4] = mario_velocity.x
            mario[5] = mario_velocity.y
            mario[6] = 1 if self._mario.invincible_ else 0
            mario[7] = 1 if self._mario.jumping_ else 0

            bullets = np.zeros(shape=(self.max_spawns, 5), dtype=np.float32)

            i = 0
            for bullet in self._bullet_bills:
                if bullet.state_ != 'active':
                    continue
                rect = bullet.rect
                bullets[i, 0] = rect.left
                bullets[i, 1] = rect.top
                bullets[i, 2] = rect.right
                bullets[i, 3] = rect.bottom
                bullets[i, 4] = bullet.velocity_
                i += 1

            stars = np.zeros(shape=(self._n_star, 4), dtype=np.float32)

            i = 0
            for star in self._stars:
                if not star.active_:
                    continue
                rect = star.rect
                stars[i, 0] = rect.left
                stars[i, 1] = rect.top
                stars[i, 2] = rect.right
                stars[i, 3] = rect.bottom
                i += 1

            return {
                "mario": mario,
                "bullet_bills": bullets,
                "stars": stars,
            }
        return None

    def _generate_info(self):
        return {"time_elapsed": self._time_elapsed}

    def _draw_status(
            self,
            text: str,
            color: Tuple[int, int, int],
    ):
        if self._font is None:
            self._font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        text = self._font.render(text, True, color)
        _, surface_height = self._status_surface.get_size()
        _, text_height = text.get_size()
        self._status_surface.fill(Color.WHITE)
        self._status_surface.blit(text, (10, ((surface_height - text_height) // 2)))

    def _step_internal(self, action: Tuple[float, float]):
        assert self._mario is not None

        dt = 1.0 / FPS

        self._timer += dt
        self._time_elapsed += dt

        if self._time_elapsed >= self.game_duration:
            self._game_state = 'cleared'
            if self.should_render:
                    self.render()
            return True

        self._mario.apply_force(action)

        if self._timer >= self._spawn_interval:
            self._timer = 0
            self._spawn_bullet_bill()

        self._spawn_interval = self.init_spawn_interval * (
                self.min_spawn_interval / self.init_spawn_interval
        ) ** (self._time_elapsed / self.max_spawn_duration)

        self._mario.tick_frame(dt=dt, screen_size=self._screen_size, platforms=self._nearby_platforms())
        for bullet in self._bullet_bills:
            bullet.tick_frame(dt=dt, screen_size=self._screen_size)

        self._mario.check_star_collision(self._stars)
        self._mario.check_bullet_collision(self._bullet_bills)

        active_bullets = []
        for bullet in self._bullet_bills:
            if bullet.state_ == 'oob':
                bullet.kill()
            else:
                active_bullets.append(bullet)
        self._bullet_bills = active_bullets

        active_stars = []
        for star in self._stars:
            if star.active_:
                active_stars.append(star)
            else:
                star.kill()
        self._stars = active_stars

        if not self._mario.active_:
            self._game_state = 'game_over'
            if self.should_render:
                self.render()
            return True

        if self.should_render:
            self.render()

        return False

