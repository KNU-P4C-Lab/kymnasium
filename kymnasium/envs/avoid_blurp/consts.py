import os
from enum import IntEnum


FPS = 60
SCALE = 3.0
TILE_SIZE = 16
GAME_TILE_SIZE = int(TILE_SIZE * SCALE)
STATUS_HEIGHT = 50
SPRITE_COLOR_KEY = (146, 144, 255)


class Ids(IntEnum):
    mario = 0
    land = 32


class Actions(IntEnum):
    noop = 0
    left = 1
    right = 2


MARIO_ZERO_SPEED_THRESHOLD = 10.0
MARIO_MAX_SPEED = 750.0
MARIO_MIN_ANIM_INTERVAL, MARIO_MAX_ANIM_INTERVAL = 0.005, 0.1
MARIO_FRICTION = 0.90
MARIO_DEFAULT_MOVING_FORCE = 75.0
MARIO_MAX_MOVING_FORCE = 100.0

BLURP_GRAVITY_MIN, BLURP_GRAVITY_MAX = 100, 800
BLURP_INTERVAL_ANIM = 0.1

PATH_SPRITE = os.path.join(os.path.dirname(__file__), 'assets', 'sprite.png')
FILE_OBJECT = '_object.csv'
FILE_BACKGROUND = '_background.csv'

OFFSET_MARIO_STAND = 0, 0
OFFSET_MARIO_WALK = [(16, 0), (32, 0), (48, 0)]
OFFSET_MARIO_DEAD = 160, 0

OFFSET_BLURP = [(0, 16), (16, 16)]

OFFSET_FLOOR = 96, 32

