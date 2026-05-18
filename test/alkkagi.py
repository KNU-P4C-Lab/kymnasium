import threading
import random
from typing import Any, Dict
import gymnasium as gym
import kymnasium as kym
from kymnasium.envs.alkkagi import ManualPlayWrapper, RemoteEnvWrapper


class RandomBlackAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def act(self, observation: Any, info: Dict):
        return {
            'turn': 0,
            'angle': random.uniform(-180, 180),
            'power': random.uniform(1, 2500),
            'index': random.choice([0, 1, 2, 3, 4])
        }


class RandomWhiteAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def act(self, observation: Any, info: Dict):
        return {
            'turn': 1,
            'angle': random.uniform(-180, 180),
            'power': random.uniform(1, 2500),
            'index': random.choice([0, 1, 2, 3, 4])
        }


def manual_play():
    wrapper = ManualPlayWrapper(
        env_id='kymnasium/AlKkaGi-3x3-v0',
        debug=False,
        render_mode='human',
        obs_type='custom',
        bgm=True
    )
    wrapper.play()


def manual_vs_agent_play():
    agent_black = RandomBlackAgent()

    wrapper = ManualPlayWrapper(
        env_id='kymnasium/AlKkaGi-3x3-v0',
        debug=False,
        agent=agent_black,
        agent_turn=0,
        render_mode='human',
        obs_type='custom',
        bgm=False
    )
    wrapper.play()

def local_random_play():
    env = gym.make(
        id='kymnasium/AlKkaGi-3x3-v0',
        render_mode='human',
        obs_type='custom',
        bgm=True
    )

    agent_black = RandomBlackAgent()
    agent_white = RandomWhiteAgent()

    done = False
    observation, info = env.reset()
    steps = 0
    while not done:
        if observation['turn'] == 0:
            action = agent_black.act(observation, info)
        else:
            action = agent_white.act(observation, info)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    return steps


def remote_random_play():
    host = "localhost"
    port = 18861

    server = RemoteEnvWrapper(
        allowed_ids=['test-1', 'test-2'],
        env_id='kymnasium/AlKkaGi-3x3-v0',
        render_mode='human',
        obs_type='custom',
        bgm=True,
        debug=True
    )

    kwargs_black = dict(
        user_id='test-1',
        agent=RandomBlackAgent(),
        host=host,
        port=port,
        debug=True
    )

    kwargs_white = dict(
        user_id='test-2',
        agent=RandomWhiteAgent(),
        host=host,
        port=port,
        debug=True
    )

    thread_black = threading.Thread(target=kym.evaluate_remote, kwargs=kwargs_black, daemon=True)
    thread_white = threading.Thread(target=kym.evaluate_remote, kwargs=kwargs_white, daemon=True)

    thread_black.start()
    thread_white.start()

    server.run(host, port)

    thread_black.join()
    thread_white.join()


def run_server(allowed_ids, host, port):
    server = RemoteEnvWrapper(
        allowed_ids=allowed_ids,
        env_id='kymnasium/AlKkaGi-3x3-v0',
        render_mode='human',
        obs_type='custom',
        bgm=True,
        debug=True
    )
    server.run(host, port)


if __name__ == "__main__":
    manual_play()




