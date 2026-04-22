import random
from typing import Any, Dict
import kymnasium as kym
from kymnasium.grid_world import ManualPlayWrapper


class RandomAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def act(self, observation: Any, info: Dict):
        action = random.choice([0, 1, 2, 3, 4, 5])
        return action


def manual_play():
    agent = ManualPlayWrapper(
        env_id='kymnasium/GridWorld-Adventure-32x32',
        render_mode='human',
        debug=True,
        bgm= True
    )
    agent.play()


def random_play():
    kym.evaluate(
        env_id='kymnasium/GridWorld-Adventure-26x26',
        agent=RandomAgent(),
        render_mode='human',
        bgm=True
    )


if __name__ == "__main__":
    manual_play()