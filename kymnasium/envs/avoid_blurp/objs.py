import pygame
import numpy as np
import math
from typing import Tuple, List
from .consts import *
from ...common.util import load_sprite
from ...common.sprite import Sprite

MarioImages = dict[str, pygame.Surface | list[pygame.Surface]]
BlurpImages = list[pygame.Surface]


class Mario(Sprite):
    def __init__(
            self,
            images: MarioImages,
            position: Tuple[int, int],
    ):
        super().__init__()

        self._walk_idx = 0
        self._walk_timer = 0
        self._active = True
        self._is_right = True
        self._is_grounded = False

        self._images = images
        self._velocity = pygame.Vector2(0, 0)

        self.image = self._images['right_stand']
        self.rect = self.image.get_rect(x=position[0], y=position[1])

    @staticmethod
    def load_images(sprite: pygame.Surface) -> MarioImages:
        images = dict()
        mapper = {
            'right_stand': (OFFSET_MARIO_STAND, False),
            'left_stand': (OFFSET_MARIO_STAND, True),
            'right_walk': (OFFSET_MARIO_WALK, False),
            'left_walk': (OFFSET_MARIO_WALK, True),
            'right_jump': (OFFSET_MARIO_JUMP, False),
            'left_jump': (OFFSET_MARIO_JUMP, True),
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
        return images

    @property
    def velocity_(self):
        return self._velocity

    @property
    def active_(self):
        return self._active

    @property
    def jumping_(self):
        return not self._is_grounded

    def set_grounded_from_platforms(self, platforms: List['WorldObjects']):
        self._is_grounded = any(
            self.rect.bottom == platform.rect.top
            and self.rect.right > platform.rect.left
            and self.rect.left < platform.rect.right
            for platform in platforms
        )

    def apply_force(self, force: Tuple[float, float]):
        fx, fy = force

        if self._is_grounded:
            fx = max(-MARIO_MAX_MOVING_FORCE, min(MARIO_MAX_MOVING_FORCE, float(fx)))
        else:
            fx = max(-MARIO_MAX_FLOATING_FORCE, min(MARIO_MAX_FLOATING_FORCE, float(fx)))
        self._velocity.x += fx

        if self._is_grounded and fy < 0:
            self._velocity.y = max(-MARIO_MAX_JUMPING_FORCE, min(0, float(fy)))
            self._is_grounded = False

    def tick_frame(self, dt: float, screen_size: Tuple[int, int], platforms: List['WorldObjects']):
        if self._is_grounded:
            self._velocity.x = max(-MARIO_GROUND_MAX_SPEED, min(MARIO_GROUND_MAX_SPEED, self._velocity.x))
        else:
            self._velocity.x = max(-MARIO_AIR_MAX_SPEED, min(MARIO_AIR_MAX_SPEED, self._velocity.x))
        self._velocity.y += MARIO_GRAVITY * dt

        if self._velocity.x > 0:
            self._is_right = True
        elif self._velocity.x < 0:
            self._is_right = False

        if abs(self._velocity.x) < MARIO_ZERO_SPEED_THRESHOLD:
            self._velocity.x = 0.0

        self.rect.x += self._velocity.x * dt
        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(screen_size[0], self.rect.right)

        self._velocity.x *= MARIO_FRICTION

        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self._velocity.x > 0:
                    self.rect.right = platform.rect.left
                elif self._velocity.x < 0:
                    self.rect.left = platform.rect.right

        self.rect.y += self._velocity.y * dt
        self._is_grounded = False

        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self._velocity.y > 0:
                    self.rect.bottom = platform.rect.top
                    self._velocity.y = 0
                    self._is_grounded = True
                elif self._velocity.y < 0:
                    self.rect.top = platform.rect.bottom
                    self._velocity.y = 0

        if self.rect.bottom > screen_size[1]:
            self._active = False

    def update(self, dt: float, *args, **kwargs):
        if not self._active:
            self.image = self._images['dead']
        elif self._is_grounded and abs(self._velocity.x) >= MARIO_ZERO_SPEED_THRESHOLD:
            self._walk_timer += dt
            speed_factor = abs(self._velocity.x) / MARIO_GROUND_MAX_SPEED
            delay = MARIO_MAX_ANIM_INTERVAL - (MARIO_MAX_ANIM_INTERVAL - MARIO_MIN_ANIM_INTERVAL) * speed_factor

            if self._walk_timer >= delay:
                self._walk_timer = 0
                self._walk_idx = (self._walk_idx + 1) % len(self._images['right_walk'])
            self.image = self._images['right_walk'][self._walk_idx] if self._is_right else self._images['left_walk'][self._walk_idx]
        elif self._is_grounded:
            self.image = self._images['right_stand'] if self._is_right else self._images['left_stand']
        else:
            self.image = self._images['right_jump'] if self._is_right else self._images['left_jump']

    def check_collision(self, blurps: List['Blurp']):
        for blurp in blurps:
            if blurp.rect.colliderect(self.rect):
                self._active = False
                break


class Blurp(Sprite):
    def __init__(
            self,
            images: BlurpImages,
            position: Tuple[float, float],
            gravity: float,
    ):
        super().__init__()

        self._gravity = gravity
        self._anim_idx = 0
        self._anim_timer = 0
        self._velocity = pygame.Vector2(0, 0)
        self._active = True

        self._images = images

        self.image = self._images[0]
        self.rect = self.image.get_rect(x=position[0], y=position[1])

    @staticmethod
    def load_images(sprite: pygame.Surface, flip: bool = False) -> BlurpImages:
        return [
            load_sprite(sprite, (TILE_SIZE, TILE_SIZE), offset, flip, scale=SCALE)
            for offset in OFFSET_BLURP
        ]

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

    def calc_velocity(self, target: Tuple[int, int] = None, angle: float = None):
        if target is not None:
            sx, sy = self.rect.center
            ex, ey = target
            dx = ex - sx
            dist = abs(dx)

            angle = angle if angle is not None else np.random.uniform(45, 85)
            rad = math.radians(angle)
            try:
                v_square = (dist * self._gravity) / math.sin(2 * rad)
                speed = math.sqrt(v_square)
            except ValueError:
                speed = 0
            direction = 1.0 if dx > 0 else -1.0
            self._velocity.x = speed * math.cos(rad) * direction
            self._velocity.y = -speed * math.sin(rad)


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
