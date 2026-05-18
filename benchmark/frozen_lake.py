import random
import pickle
import kymnasium as kym
import gymnasium as gym
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tqdm.auto import tqdm

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


def reward_func(obs, done):
    if obs == 63:
        return 1.0
    elif done:
        return -1.0
    else:
        return -0.01

@tf.function
def train_fa(model, optimizer, objective, states, actions, targets):

    with tf.GradientTape() as tape:
        # 이제 목적 함수의 값을 계산하자.
        values = model(states)

        values = values * actions

        values = keras.ops.sum(values, axis=1)

        loss = objective(targets, values)

    gradients = tape.gradient(
        loss,  # 목적 함수의 값
        model.trainable_variables  # 목적 함수를 모델의 훈련해야할 매개변수로 미분
    )
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

@tf.function
def eps_greedy(model, state):
    Q = model(keras.ops.expand_dims(state, axis=0))
    return  keras.ops.ravel(Q)


def monte_carlo_fa():
    env = gym.make(
        id='FrozenLake-v1',
        render_mode='rgb_array',
        desc=CUSTOM_MAP,
        is_slippery=False,  # 타일에서 미끄러지는 여부를 결정
    )
    # 할인율
    GAMMA = 0.99

    # 최소 Epsilon
    EPS_MIN = 0.10

    # 매 에피소드마다 감쇄시킬 Epsilon의 정도로,
    # 신경망 훈련에 시간이 좀 걸리는지라 지난 번보다 10배 크게 잡았다.
    EPS_DECAY = 0.9995

    # Epsilon을 감쇄시키지 않고 (Full Exploration) 상호작용 할 초반 에피소드의 갯수
    # 지난 번에는 15000회로 했으나, 이번에는 시간이 좀 걸리기 때문에 500회 정도로 하겠다.
    EPISODE_FULL_EXP = 5000

    # 최대로 상호작용 할 에피소드
    EPISODE_MAX = 500000

    # 조기 종료를 위한 평균 보상
    EARLY_STOP_AVG_REWARD = 0.06

    # 평균 보상을 관측할 최근 에피소드의 개수
    EPISODE_MONITOR = 10

    # 목표 지점의 좌표
    POS_GOAL = 8 * 8 - 1

    # 행동
    ACTIONS = [0, 1, 2, 3]

    # 상태의 갯수
    N_STATES = 8 * 8

    # 행동의 갯수
    N_ACTIONS = len(ACTIONS)

    # 랜덤 시드
    SEED = 42

    model = keras.models.Sequential([
        # 64차원 입력을 받는다
        keras.layers.Input(
            shape=(N_STATES,),
        ),
        # 64차원 입력을 128개의 퍼셉트론이 있는 레이어로 연결한다.
        keras.layers.Dense(
            units=128,
            activation=keras.activations.relu,
            kernel_initializer=keras.initializers.HeNormal(seed=42),
        ),
        # 4개의 퍼셉트론이 있는 레이어를 연결한다.
        # 각 퍼셉트론이 상, 하, 좌, 우 행동에 대한 행동 가치 함수를 출력하는 것이다.
        keras.layers.Dense(
            units=N_ACTIONS,
            activation=keras.activations.linear,
            kernel_initializer=keras.initializers.GlorotNormal(seed=42),
        )
    ])

    objective = keras.losses.Huber(
        # Huber Loss의 Delta로, Error가 이 값보다 작다면 Squared Error로,
        # 크다면 적절히 보정해준다. 보통은 1.0으로 잡는다.
        delta=1.0
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=0.00025,  # 학습률
        clipnorm=1.0  # Gradient Clipping
    )

    random = np.random.default_rng(SEED)
    pbar = tqdm(range(EPISODE_MAX), desc='Episode')
    recent_rewards = []

    states, actions, rewards, targets = [], [], [], []

    epsilon = 1.0
    best_reward = -1e5

    for i in pbar:
        epsilon = max(epsilon * EPS_DECAY, EPS_MIN) if i > EPISODE_FULL_EXP else epsilon

        total_reward = 0.0

        done = False
        obs, _ = env.reset()

        while not done:
            state = keras.ops.one_hot(obs, N_STATES)
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                Q = eps_greedy(model, state)
                action = np.argmax(Q)

            next_obs, _, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            reward = reward_func(next_obs, done)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs
            total_reward += reward

        G = 0.0
        for reward in reversed(rewards):
            G = reward + GAMMA * G
            targets.append(G)


        loss = train_fa(
            model, optimizer, objective,
            states=keras.ops.convert_to_tensor(states),
            actions = keras.ops.one_hot(actions, 4),
            targets = keras.ops.convert_to_tensor(targets)
        )

        # 그 다음부터는 지난 시간과 거의 비슷하다.
        if len(recent_rewards) > EPISODE_MONITOR:
            del recent_rewards[:1]

        recent_rewards.append(total_reward / len(rewards))
        avg_reward = np.mean(recent_rewards)

        if avg_reward > best_reward:
            best_reward = avg_reward

        # 훈련이 잘 되고 있는지 확인하기 위해서
        # 목적 함수의 값 또한 같이 출력해주겠다.
        pbar.set_postfix(
            eps=f'{epsilon:.5f}',
            avg_reward=f'{avg_reward:.5f}',
            best_reward=f'{best_reward:.5f}',
            loss=f'{loss:.5f}'
        )
        states.clear()
        actions.clear()
        rewards.clear()
        targets.clear()


        if best_reward > EARLY_STOP_AVG_REWARD:
            break

if __name__ == "__main__":
    monte_carlo_fa()