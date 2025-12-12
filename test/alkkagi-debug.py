import threading
import random
from typing import Any, Dict
import kymnasium as kym
from kymnasium.alkkagi import RemoteEnvWrapper
import time


class RandomBlackAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def act(self, observation: Any, info: Dict):
        if observation['turn'] != 0:
            return None

        time.sleep(random.random() * 3.0)
        action = {
            'turn': 0,
            'angle': random.uniform(-180, 180),
            'power': random.uniform(1, 2500),
            'index': random.choice([0, 1, 2, 3, 4])
        }
        print(f'''
        # Black
        - obs: {observation}
        - action: {action}
        - info: {info["steps"]}
        ''')
        return action


class RandomWhiteAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def act(self, observation: Any, info: Dict):
        if observation['turn'] != 1:
            return None
        time.sleep(random.random() * 3.0)
        action = {
            'turn': 1,
            'angle': random.uniform(-180, 180),
            'power': random.uniform(1, 2500),
            'index': random.choice([0, 1, 2, 3, 4])
        }
        print(f'''
        # White
        - obs: {observation}
        - action: {action}
        - info: {info["steps"]}
        ''')
        return action


def remote_random_play():
    host = "localhost"
    port = 18861

    server = RemoteEnvWrapper(
        allowed_ids=['black', 'white'],
        env_id='kymnasium/AlKkaGi-3x3-v0',
        render_mode='human',
        obs_type='custom',
        bgm=True,
        debug=True
    )

    kwargs_black = dict(
        user_id='black',
        agent=RandomBlackAgent(),
        host=host,
        port=port,
        debug=False
    )

    kwargs_white = dict(
        user_id='white',
        agent=RandomWhiteAgent(),
        host=host,
        port=port,
        debug=False
    )

    thread_black = threading.Thread(target=kym.evaluate_remote, kwargs=kwargs_black, daemon=True)
    thread_white = threading.Thread(target=kym.evaluate_remote, kwargs=kwargs_white, daemon=True)

    thread_black.start()
    thread_white.start()

    server.run(host, port)

    thread_black.join()
    thread_white.join()

if __name__ == "__main__":
    remote_random_play()