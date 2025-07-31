import numpy as np
from kymnasium.grid_adventure import ManualPlayWrapper
import kymnasium as kym


class RandomAgent(kym.Agent):
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def __init__(self) -> None:
        super().__init__()
        self.ran = np.random.default_rng(42)
        self.actions = [0, 1, 2, 3, 4, 5]

    def act(self, observation: any, info: dict):
        action = self.ran.choice(self.actions)
        return action


def manual():
    agent = ManualPlayWrapper(
        env_id='kymnasium/GridAdventure-FullMaze-26x26-v0',
        render_mode='human',
    )
    agent.play()


def random():
    evaluator = kym.LocalEvaluator(
        env_id='kymnasium/GridAdventure-FullMaze-26x26-v0',
        agent=RandomAgent(),
        render_mode='human',
        bgm=True
    )
    evaluator.evaluate()


if __name__ == "__main__":
    pass