from abc import ABC, abstractmethod
import pygame
import gymnasium as gym
from .util import get_logger, wait_for_close


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

    def play(self, max_play: int = 0):
        done, steps, action = True, 0, None
        play_count = 0
        running = True

        while running:
            if done:
                play_count += 1

                if 0 < max_play < play_count:
                    break

                done, steps, action = False, 0, None
                obs, info = self.env.reset()
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
                elif event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYUP and event.key == pygame.K_BACKSPACE:
                    done = True
                else:
                    action = self.handle_events(event)
                    break

            if action is None:
                action = self.default_action_

        wait_for_close(self.env)

        self._logger.info(f'Playable environment closed!')