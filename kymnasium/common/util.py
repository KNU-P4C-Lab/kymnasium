import os
import sys
from typing import Literal, Tuple, List
import gymnasium as gym
import pygame
import logging




def wait_for_close(env: gym.Env):
    if env is None:
        return

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

    env.close()


def play_bgm(path: str) -> None:
    if not os.path.exists(path):
        return

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(loops=-1)
    except pygame.error as e:
        import traceback
        traceback.print_exc()
        pass


def load_sprite(
        sprite: pygame.Surface,
        size: Tuple[int, int],
        offset: Tuple[int, int],
        flip_x: bool = False,
        flip_y: bool = False,
        scale: float = 1,
) -> pygame.Surface:
    size = (size[0] * scale, size[1] * scale)
    offset = (offset[0] * scale, offset[1] * scale)
    img = sprite.subsurface(*offset, *size)
    img = pygame.transform.flip(img, flip_x, flip_y)

    return img


def swap_colors(
        surface: pygame.Surface,
        colors: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
):
    copy = surface.copy()

    for old, new in colors:
        mask = pygame.mask.from_threshold(copy, old, threshold=(1, 1, 1, 255))
        masked_surface = mask.to_surface(setcolor=new, unsetcolor=(0, 0, 0, 0))
        copy.blit(masked_surface, (0, 0))

    return copy


def get_logger(name: str, debug: bool = False):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        formatter = logging.Formatter('[%(levelname)s - %(asctime)s - %(name)s] %(message)s')

        normal_handler = logging.StreamHandler(sys.stdout)
        normal_handler.setFormatter(formatter)
        normal_handler.setLevel(logging.DEBUG)

        error_handler = logging.StreamHandler(sys.stderr)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.WARNING)

        logger.addHandler(normal_handler)
        logger.addHandler(error_handler)

    logger.setLevel(
        logging.DEBUG if debug else logging.INFO
    )
    return logger
