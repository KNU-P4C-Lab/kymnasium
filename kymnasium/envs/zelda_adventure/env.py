from typing import Tuple, Literal
import numpy as np
import pygame
import pygame.freetype
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
        Objects.darknut: Darknut,
        Objects.goriya: Goriya,
        Objects.wizzrobe: Wizzrobe,
        Objects.armos: Armos,
        Objects.octorok: Octorok,
        Objects.tektite: Tektite,
        Objects.keese: Keese,
        Objects.moblin: Moblin,
        Objects.rope: Rope,
    }

    def __init__(
            self,
            max_steps: float,
            stage: int = 1,
            render_mode: RenderMode = None,
            obs_type: ObsType = None,
    ):
        self.render_mode = render_mode or 'human'
        self.obs_type = obs_type or 'default'
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
        self._clock = pygame.time.Clock()

        self._map = load_tile_map_data(
            os.path.join(os.path.dirname(__file__), 'assets', f'stage-{self.stage}.csv'),
            TILE_SIZE,
            self._tileset,
            SCALE
        )

        self._link_sprites = Group()
        self._tile_sprites = Group()
        self._enemy_sprites = Group()

        self._link: Link | None = None
        self._stair: Stair | None = None

        self._grid = dict()

        self.width = self._map.width
        self.height = self._map.height

        if obs_type == 'image':
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Dict(
                link=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,), dtype=np.int32
                ),
                tiles=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(len(self._map.tiles), 4), dtype=np.int32
                )
            )
        self.action_space = gym.spaces.Discrete(len(Actions))

    def render(self):
        if self.render_mode == 'none':
            return None

        dt = 1.0 / self.metadata["render_fps"]

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

        self._game_surface.fill(Color.GREY)

        self._tile_sprites.update(dt)
        self._enemy_sprites.update(dt)
        self._link_sprites.update(dt)

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
        self._stair = None
        self._grid.clear()

        self._build_stage()
        self._game_state = 'playing'

        self._steps = 0

        self.render()
        return self._generate_obs(), self._generate_info()

    def step(self, action):
        if action is None:
            self.render()
            return self._generate_obs(), 0.0, False, False, self._generate_info()

        front_obj = self._grid.get(self._link.front_position_)
        cur_obj = self._grid.get(self._link.position_)

        if action == Actions.move_forward:
            self._link.move_forward(front_obj)
        elif action == Actions.turn_left:
            self._link.turn_left()
        elif action == Actions.turn_right:
            self._link.turn_right()
        elif action == Actions.pick_up:
            sword = self._link.pick_up(cur_obj)
            if sword is not None:
                self._tile_sprites.remove(sword)
                self._grid[sword.position_] = None
        elif action == Actions.drop:
            sword = self._link.drop(cur_obj)
            if sword is not None:
                self._tile_sprites.add(sword)
                self._grid[sword.position_] = sword
        elif action == Actions.attack:
            enemy = self._link.attack(front_obj)
            if enemy is not None:
                self._enemy_sprites.remove(enemy)
                self._grid[enemy.position_] = None

        self._steps += 1

        self._link_sprites.tick_input(self._link)
        self._tile_sprites.tick_input(self._link)
        self._enemy_sprites.tick_input(self._link)

        if self._steps >= self.max_steps or self._link.status_ == LinkStatus.dead:
            self._game_state = 'game_over'
        elif self._link.status_ == LinkStatus.cleared:
            self._game_state = 'cleared'
        else:
            self._game_state = 'playing'

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

    def _build_stage(self):
        for tile in self._map.tiles:
            sprite = None
            base_obj = TILESET_BASE_TILES.get(tile.identifier)
            if base_obj is not None:
                if base_obj == Objects.water:
                    sprite = Water(tile.sprite, (tile.grid_x, tile.grid_y))
                    self._tile_sprites.add(sprite)
                elif base_obj == Objects.fire:
                    sprite = Fire(tile.sprite, (tile.grid_x, tile.grid_y))
                    self._tile_sprites.add(sprite)
                elif base_obj == Objects.stair:
                    sprite = Stair(tile.sprite, (tile.grid_x, tile.grid_y))
                    self._tile_sprites.add(sprite)
                    self._stair = sprite
                elif base_obj == Objects.wall:
                    sprite = Wall(tile.sprite, (tile.grid_x, tile.grid_y))
                    self._tile_sprites.add(sprite)
                elif base_obj == Objects.cloud:
                    sprite = Cloud(self._sprite, (tile.grid_x, tile.grid_y))
                    self._tile_sprites.add(sprite)
                elif base_obj == Objects.link:
                    self._link = Link(self._sprite, (tile.grid_x, tile.grid_y))
                    self._link_sprites.add(self._link)
            else:
                color_id, obj_id = tile.identifier // TILESET_WIDTH, tile.identifier % TILESET_WIDTH
                color, obj = TILESET_ROW_TO_COLOR[color_id], TILESET_OBJECTS[obj_id]
                if obj == Objects.sword:
                    sprite = Sword(tile.sprite, (tile.grid_x, tile.grid_y), color)
                    self._tile_sprites.add(sprite)
                else:
                    inst = self.enemy_to_inst.get(obj)
                    if inst is not None:
                        sprite = inst(self._sprite, (tile.grid_x, tile.grid_y), color)
                        self._enemy_sprites.add(sprite)

            if sprite is not None:
                self._grid[(tile.grid_x, tile.grid_y)] = sprite

    def _generate_obs(self):
        if self.obs_type == 'image':
            return self.get_frame()
        elif self.obs_type == 'default':
            link = np.array([
                self._link.position_[0],
                self._link.position_[1],
                self._link.sword_.color_ if self._link.sword_ is not None else 0,
                self._link.direction_
            ])

            tiles = []
            for (x, y), obj in self._grid.items():
                if isinstance(obj, Wall):
                    tiles.append((x, y, Objects.wall, 0))
                elif isinstance(obj, Water):
                    tiles.append((x, y, Objects.water, 0))
                elif isinstance(obj, Fire):
                    tiles.append((x, y, Objects.fire, 0))
                elif isinstance(obj, Stair):
                    tiles.append((x, y, Objects.stair, 0))
                elif isinstance(obj, Cloud):
                    tiles.append((x, y, Objects.cloud, obj.status_))
                elif isinstance(obj, Sword):
                    tiles.append((x, y, Objects.sword, obj.color_))
                elif isinstance(obj, Enemy):
                    tiles.append((x, y, obj.obj, obj.color_))

            tiles = np.array(tiles)
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
        font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        text = font.render(text, True, color)
        surface_width, surface_height = self._status_surface.get_size()
        text_width, text_height = text.get_size()
        self._status_surface.fill(Color.WHITE)
        self._status_surface.blit(text, (10, ((surface_height - text_height) // 2)))

