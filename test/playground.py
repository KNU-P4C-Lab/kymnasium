from typing import Tuple
import pygame
import kymnasium as kym

kym.Agent()

UNIT_PIXEL_SIZE = 8
UNIT_MARGIN = 2
SPRITE_COLOR_KEY = (146, 144, 255)
SCALE = 2

def _load_image(
        sprite: pygame.Surface,
        size: Tuple[int, int],
        offset: Tuple[int, int],
        scale: float = 1,
        flip: bool = False,
):
    size = (size[0] * scale, size[1] * scale)
    offset = (offset[0] * scale, offset[1] * scale)

    img = sprite.subsurface(*offset, *size)
    if flip:
        img = pygame.transform.flip(img, True, False)
    return img.convert_alpha()


class Player:
    def __init__(self, sprite: pygame.Surface):
        self.x = 0
        self.velocity = 0
        self.direction = 0

        width, height = UNIT_PIXEL_SIZE * 2, UNIT_PIXEL_SIZE * 2

        offset_x = 0
        offset_y = UNIT_PIXEL_SIZE

        self._img_standing_left = _load_image(
            sprite, (width, height),
            (offset_x, offset_y),
            scale=SCALE, flip=False,
        )

        self._img_standing_right = _load_image(
            sprite, (width, height),
            (offset_x, offset_y),
            scale=SCALE, flip=True,
        )

        offset_x += (width + UNIT_MARGIN * 2)

        self._img_walking_left, self._img_walking_right = [], []

        for _ in range(3):
            img_left = _load_image(
                sprite, (width, height),
                (offset_x, offset_y),
                scale=SCALE, flip=False,
            )
            img_right = _load_image(
                sprite, (width, height),
                (offset_x, offset_y),
                scale=SCALE, flip=True,
            )
            self._img_walking_left.append(img_left)
            self._img_walking_right.append(img_right)
            offset_x += width + UNIT_MARGIN

        offset_x += width * 2 + UNIT_MARGIN * 5
        self._img_dead = _load_image(
            sprite, (width, height),
            (offset_x, offset_y),
            scale=SCALE, flip=False,
        )
        print(self._img_dead.get_width())


class Enemy:
    def __init__(self, sprite: pygame.Surface):
        self.x = 0
        self.y = 0
        self.velocity = 0
        self.acceleration = 0

        width, height = UNIT_PIXEL_SIZE * 2, UNIT_PIXEL_SIZE * 2

        offset_x = 0
        offset_y = UNIT_PIXEL_SIZE * 19 + UNIT_MARGIN * 6

        self._img_left, self._img_right = [], []
        for _ in range(2):
            img_left = _load_image(
                sprite, (width, height),
                (offset_x, offset_y),
                scale=SCALE, flip=False,
            )
            img_right = _load_image(
                sprite, (width, height),
                (offset_x, offset_y),
                scale=SCALE, flip=True,
            )
            self._img_left.append(img_left)
            self._img_right.append(img_right)
            offset_x += width + UNIT_MARGIN



# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Main game loop
running = True
clock = pygame.time.Clock()

sprite = pygame.image.load('assets/characters.png')

sprite = pygame.transform.scale(sprite, (sprite.get_width() * 2, sprite.get_height() * 2))
sprite.set_colorkey(SPRITE_COLOR_KEY)
sprite = sprite.convert_alpha()
player = Player(sprite)

sprite = pygame.image.load('assets/enemies.png')
sprite = pygame.transform.scale(sprite, (sprite.get_width() * 2, sprite.get_height() * 2))
sprite.set_colorkey(SPRITE_COLOR_KEY)
sprite = sprite.convert_alpha()

enemy = Enemy(sprite)
idx = 0

while running:
    screen.fill((255, 255, 255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(player._img_standing_left, (0, 0))
    screen.blit(player._img_standing_right, (16 * SCALE, 0))
    screen.blit(player._img_walking_left[idx % 3], (16 * 2 * SCALE, 0))
    screen.blit(player._img_walking_right[idx % 3], (16 * 3 * SCALE, 0))
    screen.blit(player._img_dead, (16 * 4 * SCALE, 0))
    screen.blit(enemy._img_left[idx % 2], (16 * 5 * SCALE, 0))
    screen.blit(enemy._img_right[idx % 2], (16 * 6 * SCALE, 0))
    pygame.display.flip()

    clock.tick(3)
    idx += 1

pygame.quit()