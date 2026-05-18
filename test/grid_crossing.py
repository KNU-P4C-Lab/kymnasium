import random
from typing import Any, Dict
import kymnasium as kym
from kymnasium.envs.grid_world import ManualPlayWrapper


class RandomAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def act(self, observation: Any, info: Dict):
        action = random.choice([0, 1, 2])
        return action


def manual_play():
    agent = ManualPlayWrapper(
        env_id='kymnasium/GridWorld-Crossing-26x26',
        bgm=True,
        debug=True
    )
    agent.play()


def random_play():
    kym.evaluate(
        env_id='kymnasium/GridWorld-Crossing-26x26',
        agent=RandomAgent(),
        bgm=True
    )


if __name__ == "__main__":
    manual_play()
    # random_play()