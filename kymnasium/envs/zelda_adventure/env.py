from typing import Tuple, Literal
import numpy as np
import pygame
import gymnasium as gym
from .objs import Link, Cloud, Sword, Darknut, Goriya, Wizzrobe, Keese, Moblin, Rope, Octorok, Armos, Tektite, Stair, \
    Wall, Fire, Water, Enemy
from .consts import *
from ...common.sprite import load_tile_map_data, Group
from ...common.types import RenderMode, ObsType
from ...common.color import Color


class ZeldaAdventureEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'none'],
        'render_fps': FPS
    }

    enemy_to_inst = {
        Object.darknut: Darknut,
        Object.goriya: Goriya,
        Object.wizzrobe: Wizzrobe,
        Object.armos: Armos,
        Object.octorok: Octorok,
        Object.tektite: Tektite,
        Object.keese: Keese,
        Object.moblin: Moblin,
        Object.rope: Rope,
    }

    def __init__(
            self,
            max_steps: float,
            stage: int = 1,
            render_mode: RenderMode = 'human',
            obs_type: ObsType = 'default',
    ):
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.should_render = self.render_mode != 'none' or self.obs_type == 'image'
        self.stage = stage
        self.max_steps = max_steps
        self._steps = 0
        self._game_state: Literal['init', 'playing', 'game_over', 'cleared'] = 'init'

        sprite = pygame.image.load(PATH_SPRITE)
        sprite.set_colorkey(SPRITE_COLOR_KEY)
        sprite = pygame.transform.scale(
            sprite,
            (sprite.get_width() * SCALE, sprite.get_height() * SCALE)
        )
        self._sprite = sprite
        self._link_images = Link.load_images(sprite)
        self._cloud_images = Cloud.load_images(sprite)
        self._enemy_images = {}

        tileset = pygame.image.load(PATH_TILESET)
        tileset.set_colorkey(SPRITE_COLOR_KEY)
        tileset = pygame.transform.scale(
            tileset,
            (tileset.get_width() * SCALE, tileset.get_height() * SCALE)
        )
        self._tileset = tileset

        self._screen = None
        self._game_surface = None
        self._status_surface = None
        self._font = None
        self._clock = pygame.time.Clock()

        self._map = load_tile_map_data(
            str(ASSET_DIR / f'stage-{self.stage}.csv'),
            TILE_SIZE,
            self._tileset,
            SCALE
        )

        self._link_sprites = Group()
        self._static_tile_sprites = Group()
        self._tile_sprites = Group()
        self._enemy_sprites = Group()

        self._link: Link | None = None
        self._stair: Stair | None = None
        self._clouds: list[Cloud] = []

        self._grid = dict()
        self._static_tiles = []
        self._dynamic_specs = []
        self._tile_obs = np.zeros((len(self._map.tiles), 4), dtype=np.int32)
        self._tile_obs_active = np.zeros((len(self._map.tiles),), dtype=bool)
        self._tile_obs_index = {}
        self._next_tile_obs_idx = 0

        self.width = self._map.width
        self.height = self._map.height
        self._build_stage_template()

        if obs_type == 'image':
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Dict(
                link=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.int32
                ),
                tiles=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(len(self._map.tiles), 4), dtype=np.int32
                )
            )
        self.action_space = gym.spaces.Discrete(len(Action))

    def render(self):
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

        self._game_surface.fill(Color.GREY)

        self._tile_sprites.update(dt)
        self._enemy_sprites.update(dt)
        self._link_sprites.update(dt)

        self._static_tile_sprites.draw(self._game_surface)
        self._tile_sprites.draw(self._game_surface)
        self._enemy_sprites.draw(self._game_surface)
        self._link_sprites.draw(self._game_surface)

        if self._game_state == 'cleared':
            self._draw_status(f"Game Clear! Steps: {self._steps} / Dist.: {self.distance_}", Color.BLUE)
        elif self._game_state == 'game_over':
            self._draw_status(f"Game Over! Steps: {self._steps} / Dist.: {self.distance_}", Color.RED)
        else:
            self._draw_status(f"Steps: {self._steps} / Dist.: {self.distance_}", Color.BLACK)

        if self.render_mode == 'human':
            pygame.event.pump()
            self._clock.tick(self.metadata['render_fps'])
            pygame.display.flip()

        return None

    def get_frame(self):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._game_surface)), axes=(1, 0, 2)
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)

        self._link_sprites.empty()
        self._tile_sprites.empty()
        self._enemy_sprites.empty()

        self._link = None
        self._grid.clear()
        self._clouds.clear()
        self._tile_obs_active.fill(False)
        self._tile_obs_index.clear()
        self._next_tile_obs_idx = 0

        self._reset_stage()
        self._game_state = 'playing'

        self._steps = 0

        if self.render_mode != 'none' or self.obs_type == 'image':
            self.render()
        return self._generate_obs(), self._generate_info()

    def step(self, action):
        if action is None:
            if self.render_mode != 'none' or self.obs_type == 'image':
                self.render()
            return self._generate_obs(), 0.0, False, False, self._generate_info()

        front_obj = self._grid.get(self._link.front_position_)
        cur_obj = self._grid.get(self._link.position_)

        if action == Action.move_forward:
            self._link.move_forward(front_obj)
        elif action == Action.turn_left:
            self._link.turn_left()
        elif action == Action.turn_right:
            self._link.turn_right()
        elif action == Action.pick_up:
            sword = self._link.pick_up(cur_obj)
            if sword is not None:
                self._tile_sprites.remove(sword)
                self._grid[sword.position_] = None
                self._set_tile_obs_active(sword, False)
        elif action == Action.drop:
            sword = self._link.drop(cur_obj)
            if sword is not None:
                self._tile_sprites.add(sword)
                self._grid[sword.position_] = sword
                self._update_tile_obs(sword, sword.position_, Object.sword, sword.color_, True)
        elif action == Action.attack:
            enemy = self._link.attack(front_obj)
            if enemy is not None:
                self._enemy_sprites.remove(enemy)
                self._grid[enemy.position_] = None
                self._set_tile_obs_active(enemy, False)

        self._steps += 1

        for cloud in self._clouds:
            cloud.tick_input(self._link)
            self._update_tile_obs(cloud, cloud.position_, Object.cloud, cloud.status_, True)

        if self._steps >= self.max_steps or self._link.status_ == LinkStatus.dead:
            self._game_state = 'game_over'
        elif self._link.status_ == LinkStatus.cleared:
            self._game_state = 'cleared'
        else:
            self._game_state = 'playing'

        if self.render_mode != 'none' or self.obs_type == 'image':
            self.render()

        if self._game_state == 'game_over':
            return self._generate_obs(), 0.0, False, True, self._generate_info()
        elif self._game_state == 'cleared':
            return self._generate_obs(), 0.0, True, False, self._generate_info()
        else:
            return self._generate_obs(), 0.0, False, False, self._generate_info()

    def close(self):
        if self._screen is not None:
            pygame.display.quit()

        pygame.quit()

    @property
    def distance_(self):
        return (abs(self._link.position_[0] - self._stair.position_[0])
                + abs(self._link.position_[1] - self._stair.position_[1]))

    @property
    def steps_(self):
        return self._steps

    def _get_enemy_images(self, inst: type[Enemy], color: Color):
        key = (inst, color)
        images = self._enemy_images.get(key)
        if images is None:
            images = inst.load_images(self._sprite, color)
            self._enemy_images[key] = images
        return images

    def _register_tile_obs(self, obj, position: Tuple[int, int], obj_id: int, prop: int = 0):
        idx = self._next_tile_obs_idx
        self._next_tile_obs_idx += 1
        self._tile_obs_index[id(obj)] = idx
        self._tile_obs[idx] = (position[0], position[1], obj_id, prop)
        self._tile_obs_active[idx] = True

    def _update_tile_obs(self, obj, position: Tuple[int, int], obj_id: int, prop: int = 0, active: bool = True):
        idx = self._tile_obs_index[id(obj)]
        self._tile_obs[idx] = (position[0], position[1], obj_id, prop)
        self._tile_obs_active[idx] = active

    def _set_tile_obs_active(self, obj, active: bool):
        self._tile_obs_active[self._tile_obs_index[id(obj)]] = active

    def _build_stage_template(self):
        for tile in self._map.tiles:
            base_obj = TILESET_BASE_TILES.get(tile.identifier)
            if base_obj is not None:
                if base_obj == Object.water:
                    sprite = Water(tile.sprite, (tile.grid_x, tile.grid_y))
                    self._static_tiles.append((sprite, (tile.grid_x, tile.grid_y), Object.water, 0))
                    self._static_tile_sprites.add(sprite)
                elif base_obj == Object.fire:
                    sprite = Fire(tile.sprite, (tile.grid_x, tile.grid_y))
                    self._static_tiles.append((sprite, (tile.grid_x, tile.grid_y), Object.fire, 0))
                    self._static_tile_sprites.add(sprite)
                elif base_obj == Object.stair:
                    sprite = Stair(tile.sprite, (tile.grid_x, tile.grid_y))
                    self._stair = sprite
                    self._static_tiles.append((sprite, (tile.grid_x, tile.grid_y), Object.stair, 0))
                    self._static_tile_sprites.add(sprite)
                elif base_obj == Object.wall:
                    sprite = Wall(tile.sprite, (tile.grid_x, tile.grid_y))
                    self._static_tiles.append((sprite, (tile.grid_x, tile.grid_y), Object.wall, 0))
                    self._static_tile_sprites.add(sprite)
                elif base_obj == Object.cloud:
                    self._dynamic_specs.append((Object.cloud, (tile.grid_x, tile.grid_y), None, None))
                elif base_obj == Object.link:
                    self._dynamic_specs.append((Object.link, (tile.grid_x, tile.grid_y), None, None))
            else:
                color_id, obj_id = tile.identifier // TILESET_WIDTH, tile.identifier % TILESET_WIDTH
                color, obj = TILESET_ROW_TO_COLOR[color_id], TILESET_OBJECTS[obj_id]
                if obj == Object.sword:
                    self._dynamic_specs.append((Object.sword, (tile.grid_x, tile.grid_y), color, tile.sprite))
                else:
                    inst = self.enemy_to_inst.get(obj)
                    if inst is not None:
                        self._dynamic_specs.append((obj, (tile.grid_x, tile.grid_y), color, inst))

    def _reset_stage(self):
        for sprite, position, obj_id, prop in self._static_tiles:
            self._grid[position] = sprite
            self._register_tile_obs(sprite, position, obj_id, prop)

        for obj, position, color, data in self._dynamic_specs:
            if obj == Object.cloud:
                sprite = Cloud(self._cloud_images, position)
                self._tile_sprites.add(sprite)
                self._clouds.append(sprite)
                self._grid[position] = sprite
                self._register_tile_obs(sprite, position, Object.cloud, sprite.status_)
            elif obj == Object.link:
                self._link = Link(self._link_images, position)
                self._link_sprites.add(self._link)
            elif obj == Object.sword:
                sprite = Sword(data, position, color)
                self._tile_sprites.add(sprite)
                self._grid[position] = sprite
                self._register_tile_obs(sprite, position, Object.sword, color)
            else:
                inst = data
                sprite = inst(self._get_enemy_images(inst, color), position, color)
                self._enemy_sprites.add(sprite)
                self._grid[position] = sprite
                self._register_tile_obs(sprite, position, obj, color)

    def _generate_obs(self):
        if self.obs_type == 'image':
            return self.get_frame()
        elif self.obs_type == 'default':
            link = np.array([
                self._link.position_[0],
                self._link.position_[1],
                self._link.sword_.color_ if self._link.sword_ is not None else 0,
                self._link.direction_
            ], dtype=np.int32)

            tiles = self._tile_obs[self._tile_obs_active].copy()
            return {'link': link, 'tiles': tiles}
        return None

    def _generate_info(self):
        return dict(
            steps=self._steps,
            distance=self.distance_,
        )

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

