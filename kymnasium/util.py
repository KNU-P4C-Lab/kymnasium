import os
import sys
from abc import abstractmethod, ABC
from typing import Literal
import gymnasium as gym
import pygame
import logging


type ObsType = Literal['image', 'custom']


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
    except pygame.error:
        pass


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


class ManualPlayWrapper(ABC):
    def __init__(
            self,
            env_id: str,
            debug: bool = False,
            **kwargs,
    ) -> None:
        kwargs = kwargs or {}
        kwargs['render_mode'] = 'human'
        env = gym.make(env_id, **kwargs)

        if env.unwrapped.render_mode != 'human':
            raise ValueError('"render_mode" should be "human" for the manual play.')

        self.env = env
        self._logger = get_logger(env_id, debug)

    @abstractmethod
    def handle_events(self, event: pygame.event.Event):
        raise NotImplementedError()

    @property
    def default_action_(self):
        return None

    def play(self, play_once: bool = True):
        done, steps, action = True, 0, None
        play_count = 0
        running = True

        while running:
            if done:
                if play_once and play_count > 0:
                    break

                done, steps, action = False, 0, None
                obs, info = self.env.reset()
                play_count += 1
                self._logger.info(f'{play_count}th Play: {self.env.spec.id}')
                self._logger.info(f'Environment reset!')

                self._logger.debug(f'{steps}th Observation: {obs}')
                self._logger.debug(f'{steps}th Info: {info}')
            elif action is not None:
                self._logger.debug(f'{steps}th Action: {action or self.default_action_}')
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                steps += 1
                self._logger.debug(f'{steps}th Observation: {obs}')
                self._logger.debug(f'{steps}th Info: {info}')

                if done:
                    self._logger.info('Completed!')
                action = None
            else:
                self.env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                    done = True
                else:
                    action = self.handle_events(event)
                    break

            if action is None:
                action = self.default_action_

        wait_for_close(self.env)

        self._logger.info(f'Playable environment closed!')
