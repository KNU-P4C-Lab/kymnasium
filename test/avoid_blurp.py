import random
from typing import Any, Dict
import kymnasium as kym
from kymnasium.envs.avoid_blurp import ManualPlayWrapper
import gymnasium as gym
from tqdm.auto import tqdm

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
        'kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1',
        debug=True,
        render_mode='human',
        bgm=True,
        seed=42
    )
    agent.play()


def random_play():
    kym.evaluate(
        agent=RandomAgent(),
        env_id='kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1',
        debug=True,
        render_mode='human',
        bgm=True
    )


def background(n_episode=100000):
    env = gym.make(
        'kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1',
        render_mode='none',
    )

    for _ in tqdm(range(n_episode)):
        obs, _ = env.reset()
        done = False

        while not done:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            if done:
                print(obs['mario'], obs['blurps'][0])


if __name__ == "__main__":
    manual_play()
