import threading
import numpy as np
import gymnasium as gym
import kymnasium as kym
from kymnasium import RemoteEvaluator
from kymnasium.alkkagi import ManualPlayWrapper, RemoteEnvWrapper


class RandomBlackAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def __init__(self, seed: int = None):
        super().__init__()
        self.ran = np.random.default_rng(seed)

    def act(self, observation: any, info: dict):
        return {
            'turn': 0,
            'angle': self.ran.uniform(-180, 180),
            'power': self.ran.uniform(1, 2500),
            'index': None
        }


class RandomWhiteAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def __init__(self, seed: int = None):
        super().__init__()
        self.ran = np.random.default_rng(seed)

    def act(self, observation: any, info: dict):
        return {
            'turn': 1,
            'angle': self.ran.uniform(-180, 180),
            'power': self.ran.uniform(1, 2500),
            'index': None
        }


def manual():
    wrapper = ManualPlayWrapper(
        env_id='kymnasium/AlKkaGi-9x9-v0',
        debug=True,
        render_mode='human',
        obs_type='custom',
        bgm=True,
    )
    wrapper.play()


def random_local():
    env = gym.make(
        id='kymnasium/AlKkaGi-9x9-v0',
        debug=True,
        render_mode='human',
        obs_type='custom',
        bgm=True,
    )

    agent_black = RandomBlackAgent(42)
    agent_white = RandomWhiteAgent(45)

    done = False
    observation, info = env.reset()

    while not done:
        if observation['turn'] == 0:
            action = agent_black.act(observation, info)
        else:
            action = agent_white.act(observation, info)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


def random_remote_on_localhost():
    host = "localhost"
    port = 18861

    server = RemoteEnvWrapper(
        allowed_ids=['test-1', 'test-2'],
        env_id='kymnasium/AlKkaGi-9x9-v0',
        render_mode='human',
        obs_type='custom',
        bgm=True,
        debug=True
    )

    agent_black = RandomBlackAgent(42)
    agent_white = RandomWhiteAgent(45)

    client_black = RemoteEvaluator(
        eval_id='test-1',
        agent=agent_black,
        host=host,
        port=port,
        debug=True
    )

    client_white = RemoteEvaluator(
        eval_id='test-2',
        agent=agent_white,
        host=host,
        port=port,
        debug=True
    )

    thread_black = threading.Thread(target=client_black.evaluate, daemon=True)
    thread_white = threading.Thread(target=client_white.evaluate, daemon=True)

    thread_black.start()
    thread_white.start()

    server.run(host, port)

    thread_black.join()
    thread_white.join()


def run_server(host, port):
    server = RemoteEnvWrapper(
        allowed_ids=['test-1', 'test-2'],
        env_id='kymnasium/AlKkaGi-9x9-v0',
        render_mode='human',
        obs_type='custom',
        bgm=True,
        debug=True
    )

    server.run(host, port)


if __name__ == "__main__":
    manual()
    #a = dict()
    #print(dir(a))
