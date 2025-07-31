import os
import queue
import threading
import time
from typing import List

import pygame
from abc import ABC, abstractmethod
import gymnasium as gym
import rpyc
from .agent import Agent
from .util import wait_for_close, get_logger


class LocalEvaluator:
    def __init__(self, env_id: str, agent: str | Agent, debug: bool = False, **kwargs):
        kwargs = kwargs or {}
        kwargs['bgm'] = True
        env = gym.make(env_id, **kwargs)

        if env.unwrapped.render_mode != 'human':
            raise ValueError('"render_mode" should be "human" for the evaluation.')

        self.env = env
        self._logger = get_logger(env_id, debug)

        if type(agent) is str:
            if not os.path.exists(agent):
                raise ValueError(f'The agent path "{agent}" is not exists.')
            else:
                self._agent = Agent.load(agent)
        else:
            self._agent = agent

    def evaluate(self):
        done, steps = False, 0
        observation, info = self.env.reset()
        self._logger.info(f'Environment reset!')

        self._logger.debug(f'{steps}th Observation: {observation}')
        self._logger.debug(f'{steps}th Info: {info}')

        while not done:
            action = self._agent.act(observation, info)

            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self._logger.debug(f'{steps}th Action: {action}')

            steps += 1
            self._logger.debug(f'{steps}th Observation: {observation}')
            self._logger.debug(f'{steps}th Info: {info}')

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True

        self._logger.info('Completed!')

        self._logger.info('Press ESC key or click the exit button to exit.')

        wait_for_close(self.env)


class InvalidActionError(Exception):
    def __init__(self, action):
        self._action = action

    def __str__(self):
        return f"Invalid action: {self._action}"


class InvalidIdError(Exception):
    def __init__(self, eval_id):
        self._eval_id = eval_id

    def __str__(self):
        return f"Invalid evaluation id: {self._eval_id}"


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

    def on_connect(self, conn: rpyc.Connection):
        connid = conn._config['connid']
        self._logger.info(f"Client connected: {connid}")

    def on_disconnect(self, conn):
        connid = conn._config['connid']
        self._logger.info(f"Client disconnected: {connid}")

    def run(self, host: str, port: int):
        self._logger.info(f'Server initializing...')

        server = rpyc.ThreadedServer(self, hostname=host, port=18861)
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
                    self._result_queue.put(result)
                    steps += 1
                    _, _, terminated, truncated, _ = result
                    done = terminated or truncated

            if self._observation is not None:
                self.env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True

        self._logger.info('Completed!')

        if server.active:
            server.close()

        if server_thread.is_alive():
            server_thread.join()

        self._logger.info(f'Server stopped.')

        self._logger.info('Press ESC key or click the exit button to exit.')

        wait_for_close(self.env)

        self._logger.info(f'Environment closed')

    def exposed_latest_observation(self):
        return self.serialize(self._observation), self._reward, self._terminated, self._truncated, self._info

    def exposed_step(self, eval_id, action):
        self._logger.debug(f"Step requested for {eval_id}'s action: {action}")

        if not self.verify_action(action):
            self._logger.debug(f'Invalid action: {action}')
            raise InvalidActionError(action)
        elif self.allowed_ids is not None and eval_id not in self.allowed_ids:
            self._logger.debug(f'Invalid evaluation id: {eval_id}')
            raise InvalidIdError(eval_id)
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
    def serialize(self, observation: any):
        pass


class RemoteEvaluator:
    def __init__(
            self,
            eval_id: str,
            agent: str | Agent,
            host: str,
            port: int,
            max_retries: int = 5,
            delay: int = 3,
            debug: bool = False
    ) -> None:
        if type(agent) is str:
            if not os.path.exists(agent):
                raise ValueError(f'The agent path "{agent}" is not exists.')
            else:
                self._agent = Agent.load(agent)
        else:
            self._agent = agent

        self._eval_id = eval_id
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._delay = delay
        self._logger = get_logger(agent.__class__.__name__, debug)

        self._conn: rpyc.Connection | None = None
        self._bg_serv: rpyc.BgServingThread | None = None
        self._service = None
        self._cnt = 0

    def _connect(self):
        if self._conn is not None and not self._conn.closed:
            return

        self._close()

        for i in range(self._max_retries):
            try:
                self._conn = rpyc.connect(self._host, self._port, config={'connid': self._eval_id})
                break
            except ConnectionRefusedError:
                self._logger.debug(f'Connection refused in {i + 1}/{self._max_retries} trials')
            except Exception as e:
                self._logger.debug(f'Failed to connect the server in {i + 1}/{self._max_retries} trials: {e}')
            time.sleep(self._delay)

        if self._conn is not None:
            self._bg_serv = rpyc.BgServingThread(self._conn)
            self._service = self._conn.root

            self._logger.info(f'Connected to the server: {self._host}:{self._port}')
        else:
            self._logger.error(f'Failed to connect the server: {self._host}:{self._port}')
            self._close()
            raise Exception(f'Failed to connect the server: {self._host}:{self._port}')

    def _close(self):
        if self._bg_serv:
            self._bg_serv.stop()
        if self._conn:
            self._conn.close()

        self._bg_serv = None
        self._conn = None
        self._service = None

        self._logger.info(f'Disconnected from the server: {self._host}:{self._port}')

    def _call_with_retry(self, func, **kwargs):
        for i in range(self._max_retries):
            try:
                return func(**kwargs)
            except InvalidActionError as e:
                self._logger.debug(f'Failed due to the invalid action; try it later: {e}')
                return None
            except InvalidIdError as e:
                self._logger.debug(f'Failed due to the invalid evaluation id. Check id again.')
                raise e
            except Exception as e:
                self._logger.debug(f'Failed in {i + 1}/{self._max_retries} trials: {e}')

            time.sleep(self._delay)

        self._logger.error(f'Failed after {self._max_retries} trials.')
        raise ConnectionError(f'Failed after {self._max_retries} trials.')

    def evaluate(self):
        done = False
        observation, reward, info, action = None, None, None, None

        while not done:
            self._connect()
            res = self._call_with_retry(self._service.latest_observation)

            if res is not None:
                observation, reward, terminated, truncated, info = res
                done = terminated or truncated

                self._logger.debug(f'{self._cnt}th Observation: {observation}')
                self._logger.debug(f'{self._cnt}th Info: {info}')

                if not done:
                    action = self._agent.act(observation, info)

            if action is not None:
                self._logger.debug(f'{self._cnt}th Action: {action}')
                try:
                    self._call_with_retry(
                        self._service.step, eval_id=self._eval_id, action=action
                    )
                    self._logger.info(f'{self._cnt}th interaction completed!')
                    self._cnt += 1
                except ConnectionError:
                    pass

                action = None

            time.sleep(self._delay)

        self._logger.info('Completed!')

        if info:
            self._logger.info(f'Info: {info}')

        self._close()
