import pygame
from typing import Tuple, List, Literal
from .consts import *
from ...common.util import load_sprite, swap_colors
from ...common.sprite import Sprite

MarioImages = dict[str, pygame.Surface | list[pygame.Surface]]
BulletBillImages = dict[bool, pygame.Surface]
StarImages = list[pygame.Surface]


class Mario(Sprite):
    def __init__(
            self,
            images: MarioImages,
            position: Tuple[int, int]
    ):
        super().__init__()
        self._active = True
        self._is_right = True
        self._is_grounded = False
        self._is_invincible = False

        self._walk_idx = 0
        self._walk_timer = 0

        self._invincible_anim_idx = 0
        self._invincible_anim_timer = 0
        self._invincible_timer = 0

        self._velocity = pygame.Vector2(0, 0)

        self._images = images
        self.image = self._images['right_stand/0']
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
        swaps = [
            # Default mario
            None,
            # Green-pale mario
            [
                (MARIO_COLOR_SKIN, MARIO_COLOR_SKIN_WHITE),
                (MARIO_COLOR_SHIRT, MARIO_COLOR_SHIRT_GREEN),
                (MARIO_COLOR_INNER, MARIO_COLOR_INNER_ORANGE),
            ],
            # Red-pale mario
            [
                (MARIO_COLOR_SKIN, MARIO_COLOR_SKIN_WHITE),
                (MARIO_COLOR_SHIRT, MARIO_COLOR_SHIRT_RED),
                (MARIO_COLOR_INNER, MARIO_COLOR_INNER_ORANGE),
            ],
            # Black-darker mario
            [
                (MARIO_COLOR_SKIN, MARIO_COLOR_SKIN_LIGHT),
                (MARIO_COLOR_SHIRT, MARIO_COLOR_SHIRT_BLACK),
                (MARIO_COLOR_INNER, MARIO_COLOR_INNER_BROWN),
            ],
        ]
        for key, (offsets, flip_x) in mapper.items():
            size = TILE_SIZE, TILE_SIZE

            if isinstance(offsets, list):
                frames = [
                    load_sprite(sprite, size, offset, flip_x, scale=SCALE)
                    for offset in offsets
                ]
                for i, swap in enumerate(swaps):
                    if swap is None:
                        images[f'{key}/{i}'] = frames
                    else:
                        images[f'{key}/{i}'] = [swap_colors(frame, swap) for frame in frames]
            else:
                frame = load_sprite(sprite, size, offsets, flip_x, scale=SCALE)
                for i, swap in enumerate(swaps):
                    if swap is None:
                        images[f'{key}/{i}'] = frame
                    else:
                        images[f'{key}/{i}'] = swap_colors(frame, swap)

        return images

    @property
    def velocity_(self):
        return self._velocity

    @property
    def invincible_(self):
        return self._is_invincible

    @property
    def active_(self):
        return self._active

    @property
    def jumping_(self):
        return not self._is_grounded

    def apply_force(self, force: Tuple[float, float]):
        fx, fy = force

        if self._is_grounded:
            fx = max(-MARIO_MAX_MOVING_FORCE, min(MARIO_MAX_MOVING_FORCE, float(fx)))
        else:
            fx = max(-MARIO_MAX_FLOATING_FORCE, min(MARIO_MAX_FLOATING_FORCE, float(fx)))
        self._velocity.x += fx

        if self._is_grounded and fy < 0:
            self._velocity.y = max(-MARIO_JUMPING_FORCE, min(0, float(fy)))
            self._is_grounded = False

    def tick_frame(self, dt: float, screen_size: Tuple[int, int], platforms: List['WorldObjects'], *args, **kwargs):
        if self._is_grounded:
            self._velocity.x = max(-MARIO_GROUND_MAX_SPEED, min(MARIO_GROUND_MAX_SPEED, self._velocity.x))
        else:
            self._velocity.x = max(-MARIO_AIR_MAX_SPEED, min(MARIO_AIR_MAX_SPEED, self._velocity.x))
        self._velocity.y += MARIO_GRAVITY * dt

        if self._velocity.x > 0:
            self._is_right = True
        elif self._velocity.x < 0:
            self._is_right = False

        if abs(self._velocity.x) < MARIO_IDLE_THRESHOLD:
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
        self.rect.y += (self._velocity.y * dt)
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

        if self._is_invincible:
            if self._invincible_timer > 0:
                self._invincible_timer -= dt
            else:
                self._is_invincible = False
                self._invincible_timer = 0

        if self.rect.bottom > screen_size[1]:
            self._active = False

    def update(self, dt: float):
        if self._is_invincible:
            self._invincible_anim_timer += dt
            if self._invincible_anim_timer >= MARIO_INVINCIBLE_ANIM_INTERVAL:
                self._invincible_anim_timer = 0
                self._invincible_anim_idx = (self._invincible_anim_idx + 1) % MARIO_INVINCIBLE_FRAMES
        else:
            self._invincible_anim_idx = 0

        inv_idx = self._invincible_anim_idx
        if not self._active:
            self.image = self._images['dead/0']
        elif self._is_grounded:
            if abs(self._velocity.x) >= MARIO_IDLE_THRESHOLD:
                self._walk_timer += dt
                speed_factor = abs(self._velocity.x) / MARIO_GROUND_MAX_SPEED
                delay = MARIO_MAX_ANIM_INTERVAL - (MARIO_MAX_ANIM_INTERVAL - MARIO_MIN_ANIM_INTERVAL) * speed_factor
                if self._walk_timer >= delay:
                    self._walk_timer = 0
                    self._walk_idx = (self._walk_idx + 1) % len(self._images['right_walk/0'])
                self.image = self._images[f'right_walk/{inv_idx}'][self._walk_idx] if self._is_right else self._images[f'left_walk/{inv_idx}'][self._walk_idx]
            else:
                self.image = self._images[f'right_stand/{inv_idx}'] if self._is_right else self._images[
                    f'left_stand/{inv_idx}']
        elif not self._is_grounded:
            self.image = self._images[f'right_jump/{inv_idx}'] if self._is_right else self._images[f'left_jump/{inv_idx}']

    def check_platform_collision(self, platforms: List['WorldObjects']):
        self._is_grounded = False

        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self._velocity.x > 0:
                    self.rect.right = platform.rect.left
                elif self._velocity.x < 0:
                    self.rect.left = platform.rect.right

                if self._velocity.y > 0:
                    self.rect.bottom = platform.rect.top
                    self._velocity.y = 0
                    self._is_grounded = True
                elif self._velocity.y < 0:
                    self.rect.top = platform.rect.bottom
                    self._velocity.y = 0

    def check_star_collision(self, stars: List['Star']):
        for star in stars:
            if self.rect.colliderect(star.rect):
                self._is_invincible = True
                self._invincible_timer += MARIO_INVINCIBLE_TIME
                star.active_ = False
                break

    def check_bullet_collision(self, bullets: List['BulletBill']):
        for bullet in bullets:
            if self.rect.colliderect(bullet.rect):
                if bullet.state_ == 'active':
                    if self._velocity.y > 0 and self.rect.bottom < bullet.rect.centery:
                        self._velocity.y = -MARIO_HOP_FORCE
                        bullet.state_ = 'fallen'
                    elif self._is_invincible:
                        bullet.state_ = 'fallen'
                    else:
                        self._active = False


class BulletBill(Sprite):
    def __init__(
            self,
            images: BulletBillImages,
            position: Tuple[float, float],
            velocity: float,
            flip: bool = False
    ):
        super().__init__()

        self._velocity = velocity
        self._state: Literal['active', 'oob', 'fallen'] = 'active'
        self.image = images[flip]
        self.rect = self.image.get_rect(x=position[0], y=position[1])

    @staticmethod
    def load_images(sprite: pygame.Surface) -> BulletBillImages:
        return {
            False: load_sprite(sprite, (TILE_SIZE, TILE_SIZE), OFFSET_BULLET_BILL, False, scale=SCALE),
            True: load_sprite(sprite, (TILE_SIZE, TILE_SIZE), OFFSET_BULLET_BILL, True, scale=SCALE),
        }

    @property
    def velocity_(self):
        return self._velocity

    @property
    def state_(self):
        return self._state

    @state_.setter
    def state_(self, state: Literal['active', 'oob', 'fallen']):
        self._state = state

    def tick_frame(self, dt: float, screen_size: Tuple[int, int], *args, **kwargs):
        if self._state == 'active':
            self.rect.centerx += self._velocity * dt
        elif self._state == 'fallen':
            self.rect.centery += BULLET_BILL_FALLEN_SPEED * dt

        if self._state == 'active':
            if self._velocity > 0 and self.rect.left > screen_size[0]:
                self._state = 'oob'
            elif self._velocity < 0 and self.rect.right < 0:
                self._state = 'oob'
        elif self._state == 'fallen':
            if self.rect.top > screen_size[1]:
                self._state = 'oob'

    def update(self, dt: float, *args, **kwargs):
        pass


class Star(Sprite):
    def __init__(
            self,
            images: StarImages,
            position: Tuple[int, int]
    ):
        super().__init__()

        self._anim_idx = 0
        self._anim_timer = 0
        self._active = True

        self._images = images

        self.image = self._images[0]
        self.rect = self.image.get_rect(x=position[0], y=position[1])

    @staticmethod
    def load_images(sprite: pygame.Surface) -> StarImages:
        return [
            load_sprite(sprite, (TILE_SIZE, TILE_SIZE), offset, scale=SCALE)
            for offset in OFFSET_STAR
        ]

    @property
    def active_(self):
        return self._active

    @active_.setter
    def active_(self, active: bool):
        self._active = active

    def update(self, dt: float, *args, **kwargs):
        self._anim_timer += dt
        if self._anim_timer >= STAR_ANIM_INTERVAL:
            self._anim_timer = 0
            self._anim_idx = (self._anim_idx + 1) % len(self._images)
        self.image = self._images[self._anim_idx]

    def tick_frame(self, *args, **kwargs):
        pass


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
