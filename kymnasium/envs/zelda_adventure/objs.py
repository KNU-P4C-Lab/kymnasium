import pygame
from typing import Tuple
from .consts import *
from ...common.util import load_sprite, swap_colors
from ...common.sprite import TiledSprite


LinkImages = dict[tuple, list[pygame.Surface]]
EnemyImages = dict[Directions, list[pygame.Surface]]
CloudImages = list[pygame.Surface]


class ZeldaSprite(TiledSprite):
    scale = SCALE
    tile_size = TILE_SIZE


class Link(ZeldaSprite):
    def __init__(
            self,
            images: LinkImages,
            position: Tuple[int, int]
    ):
        super().__init__()

        self._status = LinkStatus.normal
        self._direction = Directions.down
        self._position = position
        self._sword: Sword | None = None

        self._idle_index = 0
        self._idle_timer = 0

        self._action_idx = 0
        self._action_timer = 0

        self._images = images
        self.image = self._images[(self._direction, self._status)][0]
        self.set_rect(
            x=self._position[0],
            y=self._position[1]
        )

    @staticmethod
    def load_images(sprite: pygame.Surface) -> LinkImages:
        images = dict()
        mapper = {
            (Directions.down, LinkStatus.normal, False): (LINK_OFFSET_DOWN_IDLE, False),
            (Directions.right, LinkStatus.normal, False): (LINK_OFFSET_RIGHT_IDLE, False),
            (Directions.left, LinkStatus.normal, False): (LINK_OFFSET_RIGHT_IDLE, True),
            (Directions.up, LinkStatus.normal, False): (LINK_OFFSET_UP_IDLE, False),
            (None, LinkStatus.picking_up, False): (LINK_OFFSET_PICK_UP, False),
            (Directions.down, LinkStatus.attacking, False): (LINK_OFFSET_DOWN_ACTION, False),
            (Directions.right, LinkStatus.attacking, False): (LINK_OFFSET_RIGHT_ACTION, False),
            (Directions.left, LinkStatus.attacking, False): (LINK_OFFSET_RIGHT_ACTION, True),
            (Directions.up, LinkStatus.attacking, False): (LINK_OFFSET_UP_ACTION, False),
            (Directions.down, LinkStatus.attacking, True): (LINK_OFFSET_DOWN_ATTACK, False),
            (Directions.right, LinkStatus.attacking, True): (LINK_OFFSET_RIGHT_ATTACK, False),
            (Directions.left, LinkStatus.attacking, True): (LINK_OFFSET_RIGHT_ATTACK, True),
            (Directions.up, LinkStatus.attacking, True): (LINK_OFFSET_UP_ATTACK, False),
        }

        for (direction, status, is_swap_color), (offsets, flip_x) in mapper.items():
            frames = []

            for offset in offsets:
                if len(offset) == 2:
                    frames.append(load_sprite(sprite, (TILE_SIZE, TILE_SIZE), offset, flip_x, scale=SCALE))
                else:
                    frames.append(load_sprite(sprite, offset[2:], offset[:2], flip_x, scale=SCALE))

            if is_swap_color:
                for color, rgb in PALETTE_SWORD.items():
                    images[(direction, status, color)] = [
                        swap_colors(frame, [((255, 255, 255), rgb)]) for frame in frames
                    ]
            else:
                images[(direction, status)] = frames

        return images

    @property
    def direction_(self):
        return self._direction

    @property
    def status_(self):
        return self._status

    @property
    def sword_(self):
        return self._sword

    @property
    def position_(self):
        return self._position

    @property
    def front_position_(self):
        cx, cy = self._position
        dx, dy = DIRECTION_TO_VECTOR[self._direction]
        return cx + dx, cy + dy

    def _stop_action(self):
        self._status = LinkStatus.normal
        self._action_idx = 0
        self._action_timer = 0

    def turn_left(self):
        self._stop_action()
        self._direction = (self._direction - 1) % len(Directions)

    def turn_right(self):
        self._stop_action()
        self._direction = (self._direction + 1) % len(Directions)

    def move_forward(self, front_obj: pygame.sprite.Sprite | None):
        self._stop_action()
        if front_obj is None or isinstance(front_obj, (Sword, Water, Fire, Stair)):
            self._position = self.front_position_
            self.set_rect(
                x=self._position[0],
                y=self._position[1]
            )
            if isinstance(front_obj, (Water, Fire)):
                self._status = LinkStatus.dead
            elif isinstance(front_obj, Stair):
                self._status = LinkStatus.cleared
        elif isinstance(front_obj, Cloud) and front_obj.passable_:
            self._position = self.front_position_
            self.set_rect(
                x=self._position[0],
                y=self._position[1]
            )

    def pick_up(self, sword: pygame.sprite.Sprite | None):
        if not isinstance(sword, Sword):
            return None

        if self._sword is None:
            self._status = LinkStatus.picking_up
            self._sword = sword

        return self._sword

    def drop(self, cur_obj: pygame.sprite.Sprite | None):
        if cur_obj is not None:
            return None

        if self._sword is not None:
            self._stop_action()
            sword = self._sword
            sword.position_ = self._position
            self._sword = None
            return sword

        return None

    def attack(self, enemy: pygame.sprite.Sprite | None) -> 'Enemy | None':
        self._status = LinkStatus.attacking

        if not isinstance(enemy, Enemy):
            return None

        enemy.take_damage(self._sword)

        if enemy.status_ == EnemyStatus.dead:
            return enemy

        return None

    def update(self, dt: float, *args, **kwargs):
        if self._status == LinkStatus.normal:
            self._idle_timer += dt
            if self._idle_timer >= DEFAULT_ANIM_INTERVAL:
                self._idle_timer = 0
                self._idle_index = (self._idle_index + 1) % len(self._images[(self._direction, self._status)])
                self.image = self._images[(self._direction, self._status)][self._idle_index]
        else:
            self._action_timer += dt

            if self._status == LinkStatus.picking_up:
                interval = DEFAULT_ANIM_INTERVAL
                frames = self._images[(None, LinkStatus.picking_up)]
            else:
                if self._sword is not None:
                    interval = ATTACK_ANIM_INTERVAL
                    frames = self._images[(self._direction, LinkStatus.attacking, self._sword.color_)]
                else:
                    interval = DEFAULT_ANIM_INTERVAL
                    frames = self._images[(self._direction, LinkStatus.attacking)]

            self.image = frames[self._action_idx]

            if self._direction == Directions.down or self._direction == Directions.right:
                self.set_rect(x = self._position[0], y = self._position[1])
            elif self._direction == Directions.left:
                self.set_rect(right=self._position[0] + 1, y=self._position[1])
            elif self._direction == Directions.up:
                self.set_rect(x=self._position[0], bottom=self._position[1] + 1)

            if self._action_timer >= interval:
                self._action_timer = 0
                self._action_idx += 1

            if self._action_idx >= len(frames):
                self._stop_action()

class Wall(ZeldaSprite):
    def __init__(
            self,
            image: pygame.Surface,
            position: Tuple[int, int],
    ):
        super().__init__()
        self.image = image
        self.set_rect(x=position[0], y=position[1])


class Water(ZeldaSprite):
    def __init__(
            self,
            image: pygame.Surface,
            position: Tuple[int, int],
    ):
        super().__init__()
        self.image = image
        self.set_rect(x=position[0], y=position[1])

class Fire(ZeldaSprite):
    def __init__(
            self,
            image: pygame.Surface,
            position: Tuple[int, int],
    ):
        super().__init__()
        self.image = image
        self.set_rect(x=position[0], y=position[1])


class Stair(ZeldaSprite):
    def __init__(
            self,
            image: pygame.Surface,
            position: Tuple[int, int],
    ):
        super().__init__()
        self._position = position
        self.image = image
        self.set_rect(x=position[0], y=position[1])

    @property
    def position_(self):
        return self._position


class Cloud(ZeldaSprite):
    def __init__(
            self,
            images: CloudImages,
            position: Tuple[int, int],
    ):
        super().__init__()

        self._position = position
        self._counter = 0

        self._images = images
        self._status = CloudStatus.disappeared
        self.image = self._images[0]
        self.set_rect(x=position[0], y=position[1])

    @staticmethod
    def load_images(sprite: pygame.Surface) -> CloudImages:
        size = TILE_SIZE, TILE_SIZE
        return [
            load_sprite(sprite, size, offset, False, scale=SCALE)
            for offset in CLOUD_OFFSET
        ]

    @property
    def status_(self):
        return self._status

    @property
    def position_(self):
        return self._position

    @property
    def passable_(self):
        return self._status in (CloudStatus.disappeared, CloudStatus.disappeared_again)

    def tick_input(self, cur_obj: pygame.sprite.Sprite | None):
        if isinstance(cur_obj, Link) and cur_obj.position_ == self._position:
            return

        self._counter += 1
        if self._counter % CLOUD_UNIT_DURATION == 0:
            self._status = (self._status + 1) % len(CloudStatus)

    def update(self, *args, **kwargs):
        if self._status == CloudStatus.disappeared or self._status == CloudStatus.disappeared_again:
            self.image.set_alpha(0)
        else:
            if self._status == CloudStatus.appearing_slightly or self._status == CloudStatus.disappearing_more:
                self.image = self._images[0]
            elif self._status == CloudStatus.appearing_more or self._status == CloudStatus.disappearing_slightly:
                self.image = self._images[1]
            else:
                self.image = self._images[2]
            self.image.set_alpha(255)


class Sword(ZeldaSprite):
    def __init__(
            self,
            image: pygame.Surface,
            position: Tuple[int, int],
            color: int,
    ):
        super().__init__()

        self._color = color
        self._position = position
        self._is_picked_up = False

        self.image = image
        self.set_rect(x=position[0], y=position[1])

    @property
    def color_(self):
        return self._color

    @property
    def position_(self):
        return self._position

    @position_.setter
    def position_(self, position: Tuple[int, int]):
        self._position = position
        self.rect = self.image.get_rect(x=position[0] * TILE_SIZE * SCALE, y=position[1] * TILE_SIZE * SCALE)

    def update(self, *args, **kwargs):
        if self._is_picked_up:
            self.image.set_alpha(0)
        else:
            self.image.set_alpha(255)


class Enemy(ZeldaSprite):
    obj: int = Object.none
    mapper: dict = None
    default_direction: int = None
    hit_points: float = None
    allow_bare_damage: bool = True
    allow_normal_damage: bool = True

    def __init__(
            self,
            images: EnemyImages,
            position: Tuple[int, int],
            color: int,
    ):
        super().__init__()

        self._status = EnemyStatus.normal
        self._direction = self.default_direction
        self._position = position
        self._color = color
        self._hit_points = self.hit_points

        self._idle_idx = 0
        self._idle_timer = 0

        self._images = images
        self.image = self._images[self._direction][self._idle_idx]
        self.set_rect(x=position[0], y=position[1])

    @classmethod
    def load_images(cls, sprite: pygame.Surface, color: int) -> EnemyImages:
        images = dict()
        for direction, (offsets, flip_x) in cls.mapper.items():
            size = TILE_SIZE, TILE_SIZE
            frames = [
                load_sprite(sprite, size, offset, flip_x, scale=SCALE)
                for offset in offsets
            ]
            palette = list(zip(PALETTE_ENEMY[Color.blue], PALETTE_ENEMY[color]))
            frames = [
                swap_colors(
                    frame, palette
                ) for frame in frames
            ]
            images[direction] = frames
        return images

    @property
    def status_(self):
        return self._status

    @property
    def position_(self):
        return self._position

    @property
    def direction_(self):
        return self._direction

    @property
    def hit_points_(self):
        return self._hit_points

    @property
    def color_(self):
        return self._color

    def take_damage(self, sword: Sword | None):
        if sword is not None:
            if sword.color_ == self._color:
                damage = LINK_ATTACK_CRITICAL_DAMAGE
            else:
                damage = LINK_ATTACK_NORMAL_DAMAGE if self.allow_normal_damage else 0.0
        else:
            damage = LINK_ATTACK_BARE_DAMAGE if self.allow_bare_damage else 0.0

        self._hit_points -= damage
        if self._hit_points <= 1e-4:
            self._status = EnemyStatus.dead

    def update(self, dt: float):
        self._idle_timer += dt
        if self._idle_timer >= DEFAULT_ANIM_INTERVAL:
            self._idle_timer = 0
            self._idle_idx = (self._idle_idx + 1) % len(self._images[self._direction])
            self.image = self._images[self._direction][self._idle_idx]


class Darknut(Enemy):
    obj = Object.darknut
    mapper = {
        Directions.down: (DARKNUT_OFFSET_DOWN, False),
        Directions.right: (DARKNUT_OFFSET_RIGHT, False),
        Directions.left: (DARKNUT_OFFSET_RIGHT, True),
        Directions.up: (DARKNUT_OFFSET_DOWN, False),
    }
    default_direction = Directions.down
    hit_points = DARKNUT_HIT_POINT
    allow_bare_damage = False


class Goriya(Enemy):
    obj = Object.goriya
    mapper = {
        Directions.down: (GORIYA_OFFSET_DOWN, False),
        Directions.right: (GORIYA_OFFSET_RIGHT, False),
        Directions.left: (GORIYA_OFFSET_RIGHT, True),
        Directions.up: (GORIYA_OFFSET_DOWN, False),
    }
    default_direction = Directions.down
    hit_points = GORIYA_HIT_POINT


class Wizzrobe(Enemy):
    obj = Object.wizzrobe
    mapper = {
        Directions.down: (WIZZROBE_OFFSET_DOWN, False),
        Directions.up: (WIZZROBE_OFFSET_UP, False)
    }
    default_direction = Directions.down
    hit_points = WIZZROBE_HIT_POINT


class Rope(Enemy):
    obj = Object.rope
    mapper = {
        Directions.right: (ROPE_OFFSET_RIGHT, False),
        Directions.left: (ROPE_OFFSET_RIGHT, True)
    }
    default_direction = Directions.right
    hit_points = ROPE_HIT_POINT


class Moblin(Enemy):
    obj = Object.moblin
    mapper = {
        Directions.down: (MOBLIN_OFFSET_DOWN, False),
        Directions.right: (MOBLIN_OFFSET_RIGHT, False),
        Directions.left: (MOBLIN_OFFSET_RIGHT, True),
        Directions.up: (MOBLIN_OFFSET_DOWN, False),
    }
    default_direction = Directions.down
    hit_points = MOBLIN_HIT_POINT


class Armos(Enemy):
    obj = Object.armos
    mapper = {
        Directions.down: (ARMOS_OFFSET_DOWN, False),
        Directions.up: (ARMOS_OFFSET_UP, False),
    }
    default_direction = Directions.down
    hit_points = ARMOS_HIT_POINT
    allow_bare_damage = False
    allow_normal_damage = False


class Octorok(Enemy):
    obj = Object.octorok
    mapper = {
        Directions.down: (OCTOROK_OFFSET, False),
    }
    default_direction = Directions.down
    hit_points = OCTOROK_HIT_POINT


class Keese(Enemy):
    obj = Object.keese
    mapper = {
        Directions.down: (KEESE_OFFSET, False),
    }
    default_direction = Directions.down
    hit_points = KEESE_HIT_POINT


class Tektite(Enemy):
    obj = Object.tektite
    mapper = {
        Directions.down: (TEKTITE_OFFSET, False),
    }
    default_direction = Directions.down
    hit_points = TEKTITE_HIT_POINT
