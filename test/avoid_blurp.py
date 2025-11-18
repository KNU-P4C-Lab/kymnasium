import random
from typing import Any, Dict
import kymnasium as kym
from kymnasium.avoid_blurp import ManualPlayWrapper


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
        'kymnasium/AvoidBlurp-Normal-v0',
        debug=True,
        render_mode='human',
        bgm=True
    )
    agent.play(play_once=True)


def random_play():
    kym.evaluate(
        agent=RandomAgent(),
        env_id='kymnasium/AvoidBlurp-Normal-v0',
        debug=True,
        render_mode='human',
        bgm=True
    )


if __name__ == "__main__":
    manual_play()
    import gymnasium as gym
