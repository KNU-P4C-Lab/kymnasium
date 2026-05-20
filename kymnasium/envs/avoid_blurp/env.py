from typing import Tuple, List, Literal
import numpy as np
import pygame
import gymnasium as gym
from .objs import Mario, Blurp, WorldObjects
from .consts import *
from ...common.color import Color
from ...common.sprite import load_tile_map_data, Group
from ...common.types import RenderMode, ObsType


class AvoidBlurpEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'none'],
        'render_fps': FPS,
    }

    def __init__(
            self,
            game_duration: float,
            init_spawn_interval: float,
            min_spawn_interval: float,
            max_spawns: int,
            stage: int = 1,
            continuous_action: bool = False,
            prob_spawn_on_player: float = 0.0,
            max_spawn_duration: float = 60.0,
            frame_skip: int = 4,
            mode: Literal['vertical', 'ballistic'] = 'vertical',
            render_mode: RenderMode = 'human',
            obs_type: ObsType = 'default',
            seed: int | None = None
    ):
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.should_render = self.render_mode != 'none' or self.obs_type == 'image'

        self.stage = stage
        self.game_duration = game_duration
        self.init_spawn_interval = init_spawn_interval
        self.min_spawn_interval = min_spawn_interval
        self.max_spawns = max_spawns
        self.max_spawn_duration = min(max_spawn_duration, game_duration) if max_spawn_duration else game_duration
        self.prob_spawn_on_player = prob_spawn_on_player
        self.frame_skip = frame_skip
        self.mode = mode
        self.continuous_action = continuous_action

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
        self._blurp_images = {
            False: Blurp.load_images(sprite, flip=False),
            True: Blurp.load_images(sprite, flip=True),
        }

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
        self._blurps: List[Blurp] = []
        self._platforms: List[WorldObjects] = []
        self._mario_spawn_position: Tuple[int, int] | None = None

        self._build_static_stage()
        self._reset_dynamic_stage()

        if self.obs_type == 'default':
            self.observation_space = gym.spaces.Dict({
                "mario": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "blurps": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_spawns, 7)),
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
        self._blurps.clear()

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
        assert self._mario is not None, "Mario object must be initialized before taking actions"

        if self.continuous_action:
            return float(action[0] * MARIO_MAX_MOVING_FORCE), -float(action[1] * MARIO_MAX_JUMPING_FORCE)

        if action == Action.left_move:
            return -MARIO_DEFAULT_MOVING_FORCE, 0.0
        if action == Action.right_move:
            return MARIO_DEFAULT_MOVING_FORCE, 0.0
        if action == Action.jump:
            return 0.0, -MARIO_DEFAULT_JUMPING_FORCE
        return 0.0, 0.0

    def _build_static_stage(self):
        for tile in self._background_map.tiles:
            background = WorldObjects(tile.sprite, (tile.x, tile.y))
            self._background_sprites.add(background)

        for tile in self._object_map.tiles:
            if tile.identifier == SpriteId.mario:
                self._mario_spawn_position = (tile.x, tile.y)
            elif tile.identifier == SpriteId.land:
                obj = WorldObjects(tile.sprite, (tile.x, tile.y))
                self._static_sprites.add(obj)
                self._platforms.append(obj)

        if self._mario_spawn_position is None:
            raise ValueError(f"stage {self.stage} does not define a Mario spawn tile")

    def _reset_dynamic_stage(self):
        assert self._mario_spawn_position is not None

        self._mario = Mario(
            images=self._mario_images,
            position=self._mario_spawn_position,
        )
        self._mario.set_grounded_from_platforms(self._platforms)
        self._character_sprites.add(self._mario)

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

    def _generate_obs(self):
        if self.obs_type == 'image':
            return self.get_frame()
        elif self.obs_type == 'default':
            assert self._mario is not None

            mario = np.empty(shape=(7,), dtype=np.float32)
            mario_rect = self._mario.rect
            mario_velocity = self._mario.velocity_
            mario[0] = mario_rect.left
            mario[1] = mario_rect.top
            mario[2] = mario_rect.right
            mario[3] = mario_rect.bottom
            mario[4] = mario_velocity.x
            mario[5] = mario_velocity.y
            mario[6] = 1 if self._mario.jumping_ else 0

            blurps = np.zeros(shape=(self.max_spawns, 7), dtype=np.float32)

            for i, blurp in enumerate(self._blurps):
                rect = blurp.rect
                velocity = blurp.velocity_
                blurps[i, 0] = rect.left
                blurps[i, 1] = rect.top
                blurps[i, 2] = rect.right
                blurps[i, 3] = rect.bottom
                blurps[i, 4] = velocity.x
                blurps[i, 5] = velocity.y
                blurps[i, 6] = blurp.gravity_

            return {
                "mario": mario,
                "blurps": blurps,
            }
        return None

    def _generate_info(self):
        return {"time_elapsed": self._time_elapsed}

    def _spawn_blurp(self):
        assert self._mario is not None

        if len(self._blurps) >= self.max_spawns:
            return

        if self._random.uniform(0.0, 1.0) < self.prob_spawn_on_player:
            target_x = self._mario.rect.left + GAME_TILE_SIZE // 2
        else:
            target_x = int(self._random.uniform(0, self.width - GAME_TILE_SIZE))
        if self.mode == 'vertical':
            flip = self._random.random() < 0.5
            blurp = Blurp(
                images=self._blurp_images[flip],
                position=(target_x, 0),
                gravity=self._random.uniform(BLURP_GRAVITY_MIN, BLURP_GRAVITY_MAX),
            )
        else:
            is_left = self._random.random() < 0.5
            x = -GAME_TILE_SIZE if is_left else self.width
            blurp = Blurp(
                images=self._blurp_images[is_left],
                position=(x, self.height),
                gravity=self._random.uniform(BLURP_GRAVITY_MIN, BLURP_GRAVITY_MAX),
            )
            blurp.calc_velocity((target_x, self.height), angle=self._random.uniform(45, 85))
        self._blurps.append(blurp)
        self._character_sprites.add(blurp)

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
            self._spawn_blurp()

        self._spawn_interval = self.init_spawn_interval * (
                self.min_spawn_interval / self.init_spawn_interval
        ) ** (self._time_elapsed / self.max_spawn_duration)

        self._mario.tick_frame(dt=dt, screen_size=self._screen_size, platforms=self._platforms)
        for blurp in self._blurps:
            blurp.tick_frame(dt=dt, screen_size=self._screen_size)

        self._mario.check_collision(self._blurps)

        active_blurps = []
        for blurp in self._blurps:
            if blurp.active_:
                active_blurps.append(blurp)
            else:
                blurp.kill()
        self._blurps = active_blurps

        if not self._mario.active_:
            self._game_state = 'game_over'

            if self.should_render:
                self.render()
            return True

        if self.should_render:
            self.render()

        return False
