import random
from typing import Any, Dict
import kymnasium as kym
import gymnasium as gym
from tqdm.auto import tqdm
from kymnasium.envs.bullet_bill import ManualPlayWrapper


class RandomContinuousAgent(kym.Agent):
    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def save(self, path: str):
        pass

    def act(self, observation: Any, info: Dict):
        moving_force = random.random() * 2 - 1
        jump_force = random.random()
        return moving_force, jump_force


class RandomDiscreteAgent(kym.Agent):
    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def save(self, path: str):
        pass

    def act(self, observation: Any, info: Dict):
        action = random.choice([0, 1, 2, 3])
        return action

def manual_play():
    agent = ManualPlayWrapper(
        'kymnasium/BulletBill-Discrete-Normal-Stage-1',
        debug=False,
        render_mode='human',
        bgm=True,
        seed=42,
    )
    agent.play(max_play=2)


def random_continuous_play():
    kym.evaluate(
        agent=RandomContinuousAgent(),
        env_id='kymnasium/BulletBill-Continuous-Normal-Stage-1',
        debug=True,
        bgm=True,
        seed=42
    )

def random_discrete_play():
    kym.evaluate(
        agent=RandomDiscreteAgent(),
        env_id='kymnasium/BulletBill-Discrete-Normal-Stage-1',
        debug=True,
        bgm=True,
        seed=42
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
    random_continuous_play()