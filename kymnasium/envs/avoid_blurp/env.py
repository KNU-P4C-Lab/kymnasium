from typing import Tuple, List, Literal
import numpy as np
import pygame
import pygame.freetype
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
            max_spawn_duration: float = None,
            frame_skip: int = 4,
            mode: Literal['vertical', 'ballistic'] = 'vertical',
            render_mode: RenderMode = None,
            obs_type: ObsType = None
    ):
        self.render_mode = render_mode or 'human'
        self.obs_type = obs_type or 'default'
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

        self._random = np.random.default_rng()

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
        self._blurps: List[Blurp] = []

        self._build_stage()

        if self.obs_type == 'default':
            self.observation_space = gym.spaces.Dict({
                "mario": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,)),
                "blurps": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_spawns, 7)),
            })
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

        if self.continuous_action:
            self.action_space = gym.spaces.Box(
                low=np.inf, high=np.inf, shape=(1,)
            )
        else:
            self.action_space = gym.spaces.Discrete(3)

    def render(self):
        if self.render_mode == 'none':
            return None

        fps = self.metadata["render_fps"]
        dt = 1.0 / fps

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
            self._clock.tick(fps)
            pygame.display.flip()
        return None

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
        self._blurps.clear()

        self._time_elapsed = 0
        self._timer = 0
        self._spawn_interval = self.init_spawn_interval

        self._build_stage()
        self._game_state = 'playing'

        self.render()

        return self._generate_obs(), self._generate_info()

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
            background = WorldObjects(tile.sprite, (tile.x, tile.y))
            self._background_sprites.add(background)

        for tile in self._object_map.tiles:
            if tile.identifier == Ids.mario:
                self._mario = Mario(
                    sprite=self._sprite,
                    position=(tile.x, tile.y),
                )
                self._character_sprites.add(self._mario)
            elif tile.identifier == Ids.land:
                obj = WorldObjects(tile.sprite, (tile.x, tile.y))
                self._static_sprites.add(obj)

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

    def _generate_obs(self):
        if self.obs_type == 'image':
            return self.get_frame()
        elif self.obs_type == 'default':
            mario = np.array(
                (*self._mario.bb_, self._mario.velocity_),
                dtype=np.float32
            )
            blurps = np.zeros(shape=(self.max_spawns, 7), dtype=np.float32)

            for i, blurp in enumerate(self._blurps):
                blurps[i] = (*blurp.bb_, *blurp.velocity_, blurp.gravity_)

            return {
                "mario": mario,
                "blurps": blurps,
            }
        return None

    def _generate_info(self):
        return {"time_elapsed": self._time_elapsed}

    def _spawn_blurp(self):
        if len(self._blurps) >= self.max_spawns:
            return

        if self._random.uniform(0.0, 1.0) < self.prob_spawn_on_player:
            target_x = self._mario.bb_[0] + GAME_TILE_SIZE // 2
        else:
            target_x = int(self._random.uniform(0, self.width - GAME_TILE_SIZE))
        if self.mode == 'vertical':
            blurp = Blurp(
                sprite=self._sprite,
                position=(target_x, 0),
                gravity=self._random.uniform(BLURP_GRAVITY_MIN, BLURP_GRAVITY_MAX),
                flip=bool(self._random.choice([True, False])),
            )
        else:
            is_left = bool(self._random.choice([True, False]))
            x = -GAME_TILE_SIZE if is_left else self.width
            blurp = Blurp(
                sprite=self._sprite,
                position=(x, self.height),
                gravity=self._random.uniform(BLURP_GRAVITY_MIN, BLURP_GRAVITY_MAX),
                flip=is_left,
            )
            blurp.calc_velocity((target_x, self.height))
        self._blurps.append(blurp)
        self._character_sprites.add(blurp)

    def _step_internal(self, action):
        fps = self.metadata["render_fps"]
        dt = 1.0 / fps

        self._timer += dt
        self._time_elapsed += dt

        if self._time_elapsed >= self.game_duration:
            self._game_state = 'cleared'
            self.render()
            return self._generate_obs(), 0.0, True, False, self._generate_info()

        if not self.continuous_action:
            if action == Actions.left:
                action = -MARIO_DEFAULT_MOVING_FORCE
            elif action == Actions.right:
                action = MARIO_DEFAULT_MOVING_FORCE
            else:
                action = 0.0

        self._mario.apply_force(action)

        if self._timer >= self._spawn_interval:
            self._timer = 0
            self._spawn_blurp()

        self._spawn_interval = self.init_spawn_interval * (
                self.min_spawn_interval / self.init_spawn_interval
        ) ** (self._time_elapsed / self.max_spawn_duration)

        self._character_sprites.tick_frame(dt=dt, screen_size=(self.width, self.height))

        self._mario.check_collision(self._blurps)

        blurps_oob = [blurp for blurp in self._blurps if not blurp.active_]

        for blurp in blurps_oob:
            blurp.kill()
            self._blurps.remove(blurp)

        if not self._mario.active_:
            self._game_state = 'game_over'

        self.render()

        if self._game_state == 'playing':
            return self._generate_obs(), 0.0, False, False, self._generate_info()
        else:
            return self._generate_obs(), 0.0, False, True, self._generate_info()



