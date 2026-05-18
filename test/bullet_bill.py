import random
from typing import Any, Dict
import kymnasium as kym
import gymnasium as gym
from tqdm.auto import tqdm
from kymnasium.envs.bullet_bill import ManualPlayWrapper


class RandomAgent(kym.Agent):
    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def save(self, path: str):
        pass

    def act(self, observation: Any, info: Dict):
        action = random.choice([0, 1, 2])
        return action


def manual_play():
    agent = ManualPlayWrapper(
        'kymnasium/BulletBill-Discrete-Normal-Stage-1',
        debug=False,
        render_mode='human',
        bgm=True
    )
    agent.play(max_play=2)


def random_play():
    kym.evaluate(
        agent=RandomAgent(),
        env_id='kymnasium/BulletBill-Discrete-Easy',
        debug=True,
        render_mode='human',
        bgm=True
    )

def background(n_episode=100000):
    env = gym.make(
        'kymnasium/BulletBill-Discrete-Normal-Stage-1',
        render_mode='none',
    )

    for _ in tqdm(range(n_episode)):
        obs, _ = env.reset()
        done = False

        while not done:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            if done:
                pass
                #print(obs['mario'], obs['bullet_bills'][0])


if __name__ == "__main__":
    manual_play()