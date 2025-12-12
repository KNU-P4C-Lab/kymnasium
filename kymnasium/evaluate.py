import queue
import threading
import time
from typing import List, Any
import pygame
from abc import ABC, abstractmethod
import gymnasium as gym
import rpyc
from .agent import Agent
from .util import wait_for_close, get_logger
from .errors import NotAllowedUserIdError, InvalidActionError, ObservationError, EnvironmentClosedError


rpyc.core.vinegar._generic_exceptions_cache["kymnasium.errors.NotAllowedUserIdError"] = NotAllowedUserIdError
rpyc.core.vinegar._generic_exceptions_cache["kymnasium.errors.InvalidActionError"] = InvalidActionError


def evaluate(env_id: str, agent: Agent, debug: bool = False, **kwargs):
    kwargs = kwargs or {}
    kwargs['render_mode'] = 'human'
    logger = get_logger(env_id, debug)

    env = gym.make(env_id, **kwargs)
    logger.info(f'Environment is generated.')

    if env.unwrapped.render_mode != 'human':
        raise ValueError('"render_mode" should be "human" for the evaluation.')

    done, steps = False, 0
    observation, info = env.reset()
    logger.info(f'Environment is reset.')

    logger.debug(f'{steps}th Observation: {observation}')
    logger.debug(f'{steps}th Info: {info}')

    while not done:
        action = agent.act(observation, info)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        logger.debug(f'{steps}th Action: {action}')

        steps += 1
        logger.debug(f'{steps}th Observation: {observation}')
        logger.debug(f'{steps}th Info: {info}')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

    logger.info('Completed. Press ESC key or click the exit button to exit.')
    wait_for_close(env)

    logger.info('Environment is closed.')


class RemoteEnvWrapper(ABC, rpyc.Service):
    def __init__(
            self,
            env_id: str,
            allowed_ids: List[str] = None,
            debug: bool = False,
            **kwargs
    ) -> None:
        kwargs = kwargs or {}
        kwargs['bgm'] = True

        env = gym.make(env_id, **kwargs)

        if env.unwrapped.render_mode != 'human':
            raise ValueError('"render_mode" should be "human" for the evaluation.')

        self.env = env
        self.allowed_ids = allowed_ids
        self._logger = get_logger(env_id, debug)

        self._observation = None
        self._reward = None
        self._info = {}
        self._terminated = False
        self._truncated = False

        self._action_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._main_thread_event = threading.Event()
        self._n_conns = 0

    def on_connect(self, conn: rpyc.Connection):
        connid = conn._config['connid']
        self._n_conns += 1
        self._logger.info(f"Client connected: {connid} / # connected clients: {self._n_conns}")

    def on_disconnect(self, conn):
        connid = conn._config['connid']
        self._n_conns -= 1
        self._logger.info(f"Client disconnected: {connid} / # connected clients: {self._n_conns}")

    def run(self, host: str, port: int):
        self._logger.info(f'Server initializing...')

        server = rpyc.ThreadedServer(
            self,
            hostname=host,
            port=18861,
            protocol_config={
                'import_custom_exceptions': True,
            }
        )
        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()

        self._logger.info(f'Server started: {host}:{port}')

        self._logger.info(f'Environment initializing...')

        self._observation, self._info = self.env.reset()
        self._reward = None
        self._terminated = False
        self._truncated = False

        self._logger.info(f'Environment reset!')

        done, steps = False, 0

        self._logger.debug(f'{steps}th Observation: {self._observation}')
        self._logger.debug(f'{steps}th Info: {self._info}')

        while not done:
            if self._main_thread_event.wait(timeout=1):
                self._main_thread_event.clear()

                try:
                    action = self._action_queue.get(timeout=10.0)
                except queue.Empty:
                    action = None

                if action is not None:
                    result = self.env.step(action)
                    self._logger.debug(f'{steps}th Action: {action}')
                    self._result_queue.put(result)
                    steps += 1
                    _, _, terminated, truncated, _ = result
                    done = terminated or truncated
                    self._logger.debug(f'{steps}th Observation: {result[0]}')
                    self._logger.debug(f'{steps}th Info: {result[-1]}')

            if self._observation is not None:
                self.env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True

        self._logger.info('Completed!')
        self._logger.info('Wait for clients to be disconnected...')

        while self._n_conns > 0:
            time.sleep(3)

        self._logger.info('All clients are disconnected.')

        if server.active:
            server.close()

        if server_thread.is_alive():
            server_thread.join()

        self._logger.info(f'Server stopped.')

        self._logger.info('Press ESC key or click the exit button to exit.')

        wait_for_close(self.env)
        self._logger.info(f'Environment closed')

    def exposed_latest_observation(self, eval_id):
        if self.allowed_ids is not None and eval_id not in self.allowed_ids:
            self._logger.info(f'Invalid evaluation id: {eval_id}')
            raise NotAllowedUserIdError(eval_id)

        return self.serialize(self._observation), self._reward, self._terminated, self._truncated, self._info

    def exposed_step(self, eval_id, action):
        self._logger.debug(f"Step requested for {eval_id}'s action: {action}")

        if not self.verify_action(action):
            self._logger.debug(f'Invalid action: {action}')
            raise InvalidActionError(action)
        elif self.allowed_ids is not None and eval_id not in self.allowed_ids:
            self._logger.info(f'Invalid evaluation id: {eval_id}')
            raise NotAllowedUserIdError(eval_id)
        else:
            try:
                self._action_queue.put(action)
                self._main_thread_event.set()

                observation, reward, terminated, truncated, info = self._result_queue.get(timeout=10.0)

                self._observation = observation
                self._reward = reward
                self._terminated = terminated
                self._truncated = truncated
                self._info = info

                return
            except Exception as e:
                self._logger.error(f"Error occurred during step: {e} / {eval_id}'s action: {action}")
                raise e

    @abstractmethod
    def verify_action(self, action) -> bool:
        pass

    @abstractmethod
    def serialize(self, observation: Any):
        pass


def evaluate_remote(
        user_id: str,
        agent: str | Agent,
        host: str,
        port: int,
        delay: int = 3,
        debug: bool = False
):
    logger = get_logger(agent.__class__.__name__, debug)
    conn, service = None, None
    done = False
    step = 0

    while not done:
        try:
            if conn is None or conn.closed:
                conn = rpyc.connect(host, port, config={'connid': user_id})
                service = conn.root
                logger.info(f'Connected to the server: {host}:{port}')

            observation, reward, terminated, truncated, info = service.latest_observation(user_id)
            done = terminated or truncated

            if observation is None:
                raise ObservationError(f'Failed to obtain observation; try again later.')
            elif done:
                raise EnvironmentClosedError(f'Environment is already closed; it will stop')
            else:
                action = agent.act(observation, info)
                logger.debug(f'{step}th Observation: {observation}')
                logger.debug(f'{step}th Info: {info}')
                logger.debug(f'{step}th Action: {action}')

            if action is not None:
                service.step(user_id, action)
                logger.debug(f'{step}th interaction completed!')
                step += 1
        except ObservationError as e:
            logger.error(e)
        except InvalidActionError:
            logger.debug(f'Failed due to the invalid action; try it later')
        except NotAllowedUserIdError:
            logger.error(f'{user_id} is not allowed; Stop and check id again.')
            break
        except EnvironmentClosedError:
            logger.info('Environment is closed.')
            break
        except ConnectionError:
            logger.error(f'Failed to connect or communicate with the server: {host}:{port}')
        except Exception as e:
            logger.error(f'Failed due to unknown reason: {e}')
        finally:
            time.sleep(delay)

    logger.info('Completed!')

    if conn is not None:
        conn.close()
        logger.info(f'Disconnected to the server: {host}:{port}')



