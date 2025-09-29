import gymnasium as gym
import kymnasium as kym
from typing import Any, Dict


'''
kymnasium.Agent를 상속하여
자신만의 에이전트를 구현
'''
class YourAgent(kym.Agent):
    def act(self, observation: Any, info: Dict):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def save(self, path: str):
        pass


def train():
    '''
    Grid Crossing 환경은 다음과 같이 생성
    '''
    env = gym.make(
        id='kymnasium/environment-id-here',
        render_mode='human', # or "rgb_array"
        bgm=True # or False
    )
    '''
    여기서부터는 이 환경에 대해서 에이전트를 훈련시키는 코드를
    자유롭게 작성
    '''
