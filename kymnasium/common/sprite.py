import pygame
import numpy as np
from dataclasses import dataclass
from itertools import product
from typing import List
from .util import load_sprite


@dataclass
class TileData:
    identifier: int
    grid_x: int
    grid_y: int
    x: int
    y: int
    sprite: pygame.Surface


@dataclass
class TiledMapData:
    tiles: List[TileData]
    grid_width: int
    grid_height: int
    width: int
    height: int


def load_tile_map_data(
        path_map: str, tile_size: int, tile_set: pygame.Surface, scale: float
) -> TiledMapData:
    scaled_tile_size = tile_size * scale
    tile_set_width = tile_set.get_width() // scaled_tile_size
    tile_map = np.loadtxt(path_map, delimiter=',').T
    map_width, map_height = tile_map.shape
    scaled_width, scaled_height = int(map_width * scaled_tile_size), int(map_height * scaled_tile_size)

    tiles = list()
    for x, y in product(range(map_width), range(map_height)):
        identifier = tile_map[x, y]
        if identifier > -1:
            offset = identifier % tile_set_width * tile_size, identifier // tile_set_width * tile_size
            scaled_x, scaled_y = x * scaled_tile_size, y * scaled_tile_size
            sprite = load_sprite(tile_set, (tile_size, tile_size), offset, scale=scale)
            tiles.append(
                TileData(
                    identifier, x, y, int(scaled_x), int(scaled_y), sprite
                )
            )

    return TiledMapData(tiles, map_width, map_height, scaled_width, scaled_height)


class Group(pygame.sprite.Group):
    def tick_frame(self, *args, **kwargs):
        for sprite in self.sprites():
            sprite.tick_frame(*args, **kwargs)

    def tick_input(self, *args, **kwargs):
        for sprite in self.sprites():
            sprite.tick_input(*args, **kwargs)


class Sprite(pygame.sprite.Sprite):
    rect: pygame.Rect
    image: pygame.Surface

    def tick_frame(self, *args, **kwargs):
        pass

    def tick_input(self, *args, **kwargs):
        pass

    @property
    def bb_(self) -> tuple[int, int, int, int]:
        return self.rect.left, self.rect.top, self.rect.right, self.rect.bottom


class TiledSprite(Sprite):
    scale: float
    tile_size: int

    def set_rect(self, **kwargs):
        kwargs = {
            k: v * self.scale * self.tile_size for k, v in kwargs.items()
        }
        self.rect = self.image.get_rect(**kwargs)