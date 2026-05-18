import pygame
import numpy as np
from typing import Tuple, List
from .consts import *
from kymnasium.common.util import load_sprite
from kymnasium.common.sprite import Sprite


class Mario(Sprite):
    def __init__(
            self,
            sprite: pygame.Surface,
            position: Tuple[int, int],
    ):
        super().__init__()

        self._walk_idx = 0
        self._walk_timer = 0
        self._active = True
        self._is_right = True

        images = dict()
        mapper = {
            'right_stand': (OFFSET_MARIO_STAND, False),
            'left_stand': (OFFSET_MARIO_STAND, True),
            'right_walk': (OFFSET_MARIO_WALK, False),
            'left_walk': (OFFSET_MARIO_WALK, True),
            'dead': (OFFSET_MARIO_DEAD, False)
        }
        for key, (offsets, flip_x) in mapper.items():
            size = TILE_SIZE, TILE_SIZE
            if isinstance(offsets, list):
                images[key] = [
                    load_sprite(sprite, size, offset, flip_x, scale=SCALE)
                    for offset in offsets
                ]
            else:
                images[key] = load_sprite(sprite, size, offsets, flip_x, scale=SCALE)
        self._images = images
        self._velocity = 0.0

        self.image = self._images['right_stand']
        self.rect = self.image.get_rect(x=position[0], y=position[1])

    @property
    def velocity_(self):
        return self._velocity

    @property
    def active_(self):
        return self._active

    def apply_force(self, force: float):
        force = np.clip(force, -MARIO_DEFAULT_MOVING_FORCE, MARIO_DEFAULT_MOVING_FORCE)
        self._velocity += force

    def tick_frame(self, dt: float, screen_size: Tuple[int, int]):
        self._velocity = np.clip(self._velocity, -MARIO_MAX_SPEED, MARIO_MAX_SPEED)

        if self._velocity > 0:
            self._is_right = True
        elif self._velocity < 0:
            self._is_right = False

        if abs(self._velocity) < MARIO_ZERO_SPEED_THRESHOLD:
            self._velocity = 0.0

        self.rect.centerx += self._velocity * dt
        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(screen_size[0], self.rect.right)

        self._velocity *= MARIO_FRICTION

    def update(self, dt: float, *args, **kwargs):
        if not self._active:
            self.image = self._images['dead']
        elif abs(self._velocity) >= MARIO_ZERO_SPEED_THRESHOLD:
            self._walk_timer += dt
            speed_factor = abs(self._velocity) / MARIO_MAX_SPEED
            delay = MARIO_MAX_ANIM_INTERVAL - (MARIO_MAX_ANIM_INTERVAL - MARIO_MIN_ANIM_INTERVAL) * speed_factor

            if self._walk_timer >= delay:
                self._walk_timer = 0
                self._walk_idx = (self._walk_idx + 1) % len(self._images['right_walk'])
            self.image = self._images['right_walk'][self._walk_idx] if self._is_right else self._images['left_walk'][self._walk_idx]
        else:
            self.image = self._images['right_stand'] if self._is_right else self._images['left_stand']

    def check_collision(self, blurps: List['Blurp']):
        for blurp in blurps:
            if blurp.rect.colliderect(self.rect):
                self._active = False
                break


class Blurp(Sprite):
    def __init__(
            self,
            sprite: pygame.Surface,
            position: Tuple[float, float],
            gravity: float,
            flip: bool = False,
    ):
        super().__init__()

        self._gravity = gravity
        self._anim_idx = 0
        self._anim_timer = 0
        self._velocity = pygame.Vector2(0, 0)
        self._active = True

        self._images = [
            load_sprite(sprite, (TILE_SIZE, TILE_SIZE), offset, flip, scale=SCALE)
            for offset in OFFSET_BLURP
        ]

        self.image = self._images[0]
        self.rect = self.image.get_rect(x=position[0], y=position[1])

    @property
    def velocity_(self):
        return self._velocity

    @property
    def gravity_(self):
        return self._gravity

    @property
    def active_(self):
        return self._active

    def tick_frame(self, dt: float, screen_size: Tuple[int, int], *args, **kwargs):
        self._velocity.y += self._gravity * dt
        self.rect.centerx += self._velocity.x * dt
        self.rect.centery += self._velocity.y * dt

        self._active = self.rect.y < screen_size[1]

    def update(self, dt: float, *args, **kwargs):
        self._anim_timer += dt
        if self._anim_timer >= BLURP_INTERVAL_ANIM:
            self._anim_timer = 0
            self._anim_idx = (self._anim_idx + 1) % len(self._images)

        self.image = self._images[self._anim_idx]

    def calc_velocity(self, target: Tuple[int, int] = None):
        if target is not None:
            sx, sy = self.rect.center
            ex, ey = target
            dx = ex - sx
            dist = abs(dx)

            angle = np.random.uniform(45, 85)
            rad = np.radians(angle)
            try:
                v_square = (dist * self._gravity) / np.sin(2 * rad)
                speed = np.sqrt(v_square)
            except ValueError:
                speed = 0
            direction = 1.0 if dx > 0 else -1.0
            self._velocity.x = speed * np.cos(rad) * direction
            self._velocity.y = -speed * np.sin(rad)


class WorldObjects(Sprite):
    def __init__(
            self,
            image: pygame.Surface,
            position: Tuple[int, int],
    ):
        super().__init__()

        self.image = image
        self.rect = self.image.get_rect(x=position[0], y=position[1])

    def update(self, *args, **kwargs):
        pass

    def tick_frame(self, *args, **kwargs):
        pass