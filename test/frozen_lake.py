import random
import pickle
import kymnasium as kym
import gymnasium as gym
import numpy as np


CUSTOM_MAP = [
    "SFFFFFFH",
    "FFHFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

CUSTOM_MAP = np.array(CUSTOM_MAP, dtype='c')


# 각 행동을 취했을 때 좌표의 변화를 기록
COORDS = {
    0: (0, -1),  # 좌
    1: (1, 0),  # 하
    2: (0, 1),  # 우
    3: (-1, 0)  # 상
}


class RandomAgent(kym.Agent):
    ACTIONS = [0, 1, 2, 3]

    # 0, 1, 2, 3의 행동 중 무작위로 선택
    def act(self, obs, info):
        return random.choice(self.ACTIONS)

    # 파이썬의 Pickle 모듈로 에이전트를 저장
    def save(self, path):
        with open(path, mode='wb') as f:
            pickle.dump(self, f)

    # 파이썬의 Pickle 모듈로 에이전트를 로드
    @classmethod
    def load(cls, path: str):
        with open(path, mode='rb') as f:
            return pickle.load(f)


class TrainedAgent(kym.Agent):
    def __init__(self, PI: list):
        self.PI = PI

    # 훈련된 정책에서 행동을 선택
    def act(self, obs, info):
        return self.PI[obs]

    # 파이썬의 Pickle 모듈로 에이전트를 저장
    def save(self, path):
        with open(path, mode='wb') as f:
            pickle.dump(self, f)

    # 파이썬의 Pickle 모듈로 에이전트를 로드
    @classmethod
    def load(cls, path: str):
        with open(path, mode='rb') as f:
            return pickle.load(f)


def estimate_value(state, action, gamma, V):
    # 현재 상태로부터 행/열 번호를 추출
    row, col = state // 8, state % 8

    # 행동을 취했을 때 이동하는 곳의 좌표를 계산
    (d_row, d_col) = COORDS[action]
    new_row, new_col = row + d_row, col + d_col

    # 0 ~ 7 사이의 값으로 행/열번호 보정
    new_row = min(max(new_row, 0), 7)
    new_col = min(max(new_col, 0), 7)

    # 새로운 좌표로부터 새로운 상태 계산
    new_state = new_row * 8 + new_col

    # 이동된 좌표의 타일 정보를 추출
    tile = CUSTOM_MAP[new_row, new_col]

    # 얼음 구멍 타일의 경우 보상을 -1로 책정
    if tile == b'H':
        reward = -1
    # 도착 지점에 도착할 시 보상을 +1로 책정
    elif tile == b'G':
        reward = 1
    # 그 외 타일 (일반 타일, 시작 지점)은 보상을 0으로 책정
    else:
        reward = 0

    # 이 환경에서는 주어진 상태에서 같은 행동을 취하면
    # 항상 같은 보상 획득 및 상태 전이가 일어나므로,
    # 전이 확률 P(s',r|s, a)는 1과 같음
    # 벨만 방정식 계산
    v = reward + gamma * V[new_state]

    return v


def value_iteration(phi: float, gamma: float):
    # 상태 가치 함수를 초기화
    V = [0 for _ in range(8 * 8)]

    # 상태 가치 함수가 수렴할 때까지 업데이트
    while True:
        delta = 0
        for state in range(8 * 8):
            prev_v = V[state]

            max_v = 0
            for action in [0, 1, 2, 3]:
                v = estimate_value(state, action, gamma, V)
                max_v = max(max_v, v)

            V[state] = max_v
            delta = max(delta, abs(V[state] - prev_v))

        if delta < phi:
            break

    PI = [-1 for _ in range(8 * 8)]

    # 각 상태별로 최적의 행동을 저장
    for state in range(8 * 8):
        max_v = -1e10
        opt_action = -1

        for action in [0, 1, 2, 3]:
            v = estimate_value(state, action, gamma, V)
            if max_v < v:
                max_v = v
                opt_action = action
        PI[state] = opt_action
    return PI


def policy_iteration(phi: float, gamma: float):
    # 상태 가치 함수를 초기화
    V = [0 for _ in range(8 * 8)]

    # 정책을 초기화
    PI = [random.choice([0, 1, 2, 3]) for _ in range(8 * 8)]
    PI_prime = [None for _ in range(8 * 8)]

    # 정책이 수렴할 때까지 업데이트
    while PI != PI_prime:
        # 현재 정책에 대한 상태 가치 함수를 초기화
        for state in range(8 * 8):
            V[state] = 0

        # Policy Evaluation: 현재 정책에 대한 상태 가치 함수를 추정
        while True:
            delta = 0

            for state in range(8 * 8):
                prev_v = V[state]
                V[state] = estimate_value(state, PI[state], gamma, V)
                delta = max(delta, abs(V[state] - prev_v))

            if delta < phi:
                break

        PI_prime = list(PI)

        # Policy Improvement: 현재 정책을 더 나은 정책으로 개선
        for state in range(8 * 8):
            max_v = -1e10
            opt_action = -1

            for action in [0, 1, 2, 3]:
                v = estimate_value(state, action, gamma, V)
                if max_v < v:
                    max_v = v
                    opt_action = action

            PI[state] = opt_action
    return PI


def train(phi: float, gamma: float, is_value_iter: bool, path: str):
    if is_value_iter:
        PI = value_iteration(phi, gamma)
    else:
        PI = policy_iteration(phi, gamma)
    agent = TrainedAgent(PI)
    agent.save(path)


def run(path: str):
    agent = TrainedAgent.load(path)

    env = gym.make(
        id='FrozenLake-v1',
        render_mode='human',
        desc=CUSTOM_MAP,
        is_slippery=False,
    )

    env.reset()
    done = False

    obs, info = env.reset()

    while not done:
        # 환경의 상태와 정보를 활용하여 행동을 선택
        action = agent.act(obs, info)

        # 선택된 행동으로 환경과 상호작용하고,
        # 그로 인해 변화된 상태와 보상 등을 획득
        obs, reward, terminated, truncated, info = env.step(action)

        # 종료 여부를 확인
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    train(phi=1e-3, gamma=0.99, is_value_iter=True, path='./files/value_agent.pkl')
    kym.evaluate(
        env_id='FrozenLake-v1',
        render_mode='human',
        desc=CUSTOM_MAP,
        is_slippery=False,
        agent=TrainedAgent.load('./files/value_agent.pkl'),
    )

    train(phi=1e-3, gamma=0.99, is_value_iter=False, path='./files/policy_agent.pkl')
    kym.evaluate(
        env_id='FrozenLake-v1',
        render_mode='human',
        desc=CUSTOM_MAP,
        is_slippery=False,
        agent=TrainedAgent.load('./files/policy_agent.pkl'),
    )

