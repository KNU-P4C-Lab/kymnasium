import numpy as np
from kymnasium.avoid_blurp import ManualPlayWrapper
import kymnasium as kym


class RandomAgent(kym.Agent):
    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def save(self, path: str):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.ran = np.random.default_rng(42)

    def act(self, observation: any, info: dict):
        action = self.ran.choice([0, 1, 2])
        return action


def manual():
    agent = ManualPlayWrapper(
        'kymnasium/AvoidBlurp-Normal-v0',
        debug=True,
        render_mode='human',
        bgm=True
    )
    agent.play(play_once=True)


def random():
    evaluator = kym.LocalEvaluator(
        agent=RandomAgent(),
        env_id='kymnasium/AvoidBlurp-Normal-v0',
        debug=True,
        render_mode='human',
        bgm=True,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    manual()