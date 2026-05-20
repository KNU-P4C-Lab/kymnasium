from enum import IntEnum
from pathlib import Path

FPS = 15
SCALE = 1.5
TILE_SIZE = 16
GAME_TILE_SIZE = int(TILE_SIZE * SCALE)
STATUS_HEIGHT = 50
SPRITE_COLOR_KEY = (116, 116, 116)
TILESET_WIDTH = 10


class Color(IntEnum):
    blue = 1
    red = 2
    green = 3
    emerald = 4
    purple = 5
    yellow = 6


class Object(IntEnum):
    none = -1
    wall = 0
    water = 1
    fire = 2
    link = 3
    stair = 4
    cloud = 5
    sword = 6
    darknut = 7
    goriya = 8
    wizzrobe = 9
    rope = 10
    keese = 11
    moblin = 12
    armos = 13
    tektite = 14
    octorok = 15


class Action(IntEnum):
    stay = 0
    turn_left = 1
    turn_right = 2
    move_forward = 3
    pick_up = 4
    drop = 5
    attack = 6


class Directions(IntEnum):
    left = 0
    up = 1
    right = 2
    down = 3


class LinkStatus(IntEnum):
    dead = -1
    normal = 0
    picking_up = 1
    attacking = 2
    cleared = 3


class EnemyStatus(IntEnum):
    dead = -1
    normal = 0
    attacking = 1


class CloudStatus(IntEnum):
    disappeared = 0
    appearing_slightly = 1
    appearing_more = 2
    appearing_fully = 3
    disappearing_slightly = 4
    disappearing_more = 5
    disappeared_again = 6


TILESET_ROW_TO_COLOR = {
    2: Color.blue,
    3: Color.red,
    4: Color.green,
    5: Color.emerald,
    6: Color.purple,
    7: Color.yellow
}

TILESET_BASE_TILES = {
    -1: Object.none,
    0: Object.wall,
    1: Object.water,
    2: Object.water,
    3: Object.water,
    4: Object.water,
    5: Object.water,
    6: Object.water,
    7: Object.water,
    8: Object.water,
    9: Object.water,
    10: Object.link,
    11: Object.stair,
    12: Object.cloud,
    13: Object.fire
}

TILESET_OBJECT_START_ID = 20

TILESET_OBJECTS = {
    0: Object.sword,
    1: Object.darknut,
    2: Object.goriya,
    3: Object.wizzrobe,
    4: Object.rope,
    5: Object.keese,
    6: Object.moblin,
    7: Object.armos,
    8: Object.tektite,
    9: Object.octorok,
}

PALETTE_ENEMY = {
    Color.blue: [
        (92, 148, 252),
        (0, 0, 168)
    ],
    Color.red: [
        (252, 152, 56),
        (168, 0, 0)
    ],
    Color.green: [
        (0, 168, 0),
        (0, 80, 0)
    ],
    Color.emerald: [
        (88, 248, 152),
        (0, 60, 20)
    ],
    Color.purple: [
        (200, 147, 222),
        (187, 60, 240)
    ],
    Color.yellow: [
        (240, 188, 60),
        (64, 44, 0)
    ]
}

PALETTE_SWORD = {
    Color.blue: (92, 148, 152),
    Color.red: (216, 40, 0),
    Color.green: (0, 80, 0),
    Color.emerald: (0, 232, 216),
    Color.purple: (187, 60, 240),
    Color.yellow: (170, 117, 0)
}

ASSET_DIR = Path(__file__).parent / 'assets'
PATH_SPRITE = ASSET_DIR / 'sprite.png'
PATH_TILESET = ASSET_DIR / 'tilesets.png'

LINK_ATTACK_CRITICAL_DAMAGE = 1.0
LINK_ATTACK_NORMAL_DAMAGE = 0.25
LINK_ATTACK_BARE_DAMAGE = 0.1

DEFAULT_ANIM_INTERVAL = 0.3
ATTACK_ANIM_INTERVAL = 0.05

LINK_OFFSET_DOWN_IDLE = [(0, 0), (16, 0)]
LINK_OFFSET_RIGHT_IDLE = [(32, 0), (48, 0)]
LINK_OFFSET_UP_IDLE = [(64, 0), (80, 0)]

LINK_OFFSET_DOWN_ACTION = [(96, 0)]
LINK_OFFSET_RIGHT_ACTION = [(112, 0)]
LINK_OFFSET_UP_ACTION = [(128, 0)]

LINK_OFFSET_PICK_UP = [(144, 0), (160, 0)]

LINK_OFFSET_DOWN_ATTACK = [
    (0, 16), (16, 16, 16, 27), (32, 16, 16, 23), (48, 16, 16, 19)
]

LINK_OFFSET_RIGHT_ATTACK = [
    (64, 16), (80, 16, 27, 16), (107, 16, 23, 16), (130, 16, 19, 16)
]

LINK_OFFSET_UP_ATTACK = [
    (149, 28), (165, 16, 16, 28), (181, 17, 16, 27), (197, 25, 16, 19)
]

DARKNUT_OFFSET_DOWN = [(0, 48), (16, 48)]
DARKNUT_OFFSET_UP = [(32, 48), (48, 48)]
DARKNUT_OFFSET_RIGHT = [(64, 48), (80, 48)]
DARKNUT_HIT_POINT = 5

GORIYA_OFFSET_DOWN = [(96, 48), (112, 48)]
GORIYA_OFFSET_UP = [(128, 48), (144, 48)]
GORIYA_OFFSET_RIGHT = [(160, 48), (176, 48)]
GORIYA_HIT_POINT = 3

WIZZROBE_OFFSET_DOWN = [(0, 64), (16, 64)]
WIZZROBE_OFFSET_UP = [(32, 64), (48, 64)]
WIZZROBE_HIT_POINT = 1

ROPE_OFFSET_RIGHT = [(64, 64), (80, 64)]
ROPE_HIT_POINT = 1

MOBLIN_OFFSET_DOWN = [(96, 64), (112, 64)]
MOBLIN_OFFSET_UP = [(128, 64), (144, 64)]
MOBLIN_OFFSET_RIGHT = [(160, 64), (176, 64)]
MOBLIN_HIT_POINT = 3

ARMOS_OFFSET_DOWN = [(0, 80), (16, 80)]
ARMOS_OFFSET_UP = [(32, 80), (48, 80)]
ARMOS_HIT_POINT = 15

OCTOROK_OFFSET = [(64, 80), (80, 80)]
OCTOROK_HIT_POINT = 1

KEESE_OFFSET = [(96, 80), (112, 80)]
KEESE_HIT_POINT = 1

TEKTITE_OFFSET = [(128, 80), (144, 80)]
TEKTITE_HIT_POINT = 1

CLOUD_OFFSET = [(192, 80), (176, 80), (160, 80)]
CLOUD_UNIT_DURATION = 2

DIRECTION_TO_VECTOR = {
    Directions.left: (-1, 0),
    Directions.right: (1, 0),
    Directions.up: (0, -1),
    Directions.down: (0, 1)
}

