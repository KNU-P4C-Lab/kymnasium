import tensorflow as tf
from tensorflow import keras
import kymnasium as kym
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from tqdm.auto import tqdm
from typing import Any, Dict


def evaluate(name: str, agent: kym.Agent, env: gym.Env, seed: int = None):
    recoder = gym.wrappers.RecordVideo(
        env=env,
        video_folder='./video',
        name_prefix=name
    )

    done = False
    obs, info = recoder.reset(seed=seed)

    while not done:
        action = agent.act(obs, info)
        obs, _, terminated, truncated, info = recoder.step(action)
        done = terminated or truncated

    recoder.close()

# 할인율
GAMMA = 0.99

# 최대로 상호작용 할 에피소드
EPISODE_MAX = 20000

# 평균 보상을 관측할 최근 에피소드의 개수
EPISODE_MONITOR = 50

# 상태의 차원
DIM_STATES = 4

# 행동의 범위
ACTION_SCALE = 3.0

# 랜덤 시드
SEED = 42

# 최대 행동 횟수
MAX_STEPS = 500

# 학습률
LEARNING_RATE = 0.00001

keras.utils.set_random_seed(SEED)

class InvertedPendulumAgent(kym.Agent):
    def __init__(
            self,
            model: keras.models.Model,
    ):
        self._model = model

    def save(self, path: str):
        keras.models.save_model(
            model=self._model,
            filepath=f'{path}.keras'
        )

    @classmethod
    def load(cls, path: str):
        model = keras.models.load_model(
            filepath=f'{path}.keras'
        )
        return InvertedPendulumAgent(model=model)

    def act(self, observation: Any, info: Dict):
        state = keras.ops.expand_dims(observation, axis=0)

        # 바로 전에는 행동을 확률 분포에서 추출했지만,
        # 학습이 완료되었다면 확률적 정책 대신 결정적 정책을 써야하므로
        # 평균을 그대로 행동으로 사용하겠다.
        mu, _ = keras.ops.ravel(self._model(state))
        action = ACTION_SCALE * keras.ops.tanh(mu)
        action = np.array([action])
        return action

class RewardTracker:
    def __init__(self, monitor: int):
        self._monitor = monitor
        self._rewards = []

    def update(self, reward):
        if len(self._rewards) > self._monitor:
            del self._rewards[:1]

        self._rewards.append(reward)

    @property
    def avg_(self):
        return np.mean(self._rewards)

def build_actor():
    input = keras.layers.Input(
        shape=(DIM_STATES,),
    )

    layer = keras.layers.Dense(
        units=256,
        activation=keras.activations.relu,
        kernel_initializer=keras.initializers.HeNormal(seed=42)
    )(input)

    layer = keras.layers.Dense(
        units=256,
        activation=keras.activations.relu,
        kernel_initializer=keras.initializers.HeNormal(seed=42)
    )(layer)

    # 평균을 출력
    mu = keras.layers.Dense(
        units=1,
        activation=keras.activations.linear,
        kernel_initializer=keras.initializers.GlorotNormal(seed=42)
    )(layer)

    # 로그 표준 편차를 출력
    log_sigma = keras.layers.Dense(
        units=1,
        activation=keras.activations.linear,
        kernel_initializer=keras.initializers.GlorotNormal(seed=42)
    )(layer)

    actor = keras.models.Model(
        inputs=input,
        outputs=[mu, log_sigma]
    )
    return actor


def build_critic():
    critic = keras.models.Sequential([
        keras.layers.Input(
            shape=(DIM_STATES,)
        ),
        keras.layers.Dense(
            units=256,
            activation=keras.activations.relu,
            kernel_initializer=keras.initializers.HeNormal(seed=42),
        ),
        keras.layers.Dense(
            units=256,
            activation=keras.activations.relu,
            kernel_initializer=keras.initializers.HeNormal(seed=42),
        ),
        keras.layers.Dense(
            units=1,
            activation=keras.activations.linear,
            kernel_initializer=keras.initializers.GlorotNormal(seed=42),
        )
    ])
    return critic


objective = keras.losses.Huber()

actor_optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    clipnorm=1.0
)

critic_optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE * 5,
    clipnorm=1.0
)

actor = build_actor()
critic = build_critic()

@tf.function
def select_action(state, seed):
    mu, log_sigma = actor(state)
    mu, log_sigma = keras.ops.ravel(mu), keras.ops.ravel(log_sigma)
    sigma = keras.ops.exp(keras.ops.clip(log_sigma, -20, 2))
    x = mu + sigma * keras.random.normal(shape=(1, ), seed=seed)
    action = ACTION_SCALE * keras.ops.tanh(x)
    return action


objective = keras.losses.Huber()


@tf.function
def train(
        states,
        actions,
        returns,
        steps
):
    states = states[:steps]
    actions = actions[:steps]
    returns = returns[:steps]

    for _ in keras.ops.arange(5):
        with tf.GradientTape() as critic_tape:
            values = keras.ops.ravel(critic(states))
            critic_loss = objective(returns, values)

        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    values = keras.ops.ravel(critic(states))
    error = returns - values
    error = (error - keras.ops.mean(error)) / (keras.ops.std(error) + 1e-8)

    with tf.GradientTape() as actor_tape:
        mu, log_sigma = actor(states)
        mu, log_sigma = keras.ops.ravel(mu), keras.ops.ravel(log_sigma)

        # 지수 연산으로 로그 표준 편차를 표준 편차로 바꾸기 전에
        # 너무 값이 커지지 않도록 제약한다.
        log_sigma = keras.ops.clip(log_sigma, -20, 2)
        sigma = keras.ops.exp(log_sigma)

        # 역함수의 값을 구하기전에, 값이 커지지 않도록 제한한다.
        actions_clipped = keras.ops.clip(
            keras.ops.ravel(actions) / ACTION_SCALE, -.999999, .999999
        )
        inverse = keras.ops.arctanh(actions_clipped)

        # 로그 표준 편차를 구한다.
        log_prob_gaussian = -0.5 * keras.ops.log(2 * np.pi) - log_sigma - 0.5 * keras.ops.square(
            (inverse - mu) / (sigma + 1e-8))

        # 행동 범위에 따른 보정 값을 구한다.
        correction_term = keras.ops.log(ACTION_SCALE) + keras.ops.log(1 - keras.ops.square(actions_clipped) + 1e-8)

        log_probs = log_prob_gaussian - correction_term

        actor_loss = -keras.ops.mean(log_probs * error)

    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    return critic_loss, actor_loss

def run():
    env = gym.make(
        id='InvertedPendulum-v5',
        render_mode='rgb_array'
    )

    env = TimeLimit(env, max_episode_steps=MAX_STEPS)

    pbar = tqdm(range(EPISODE_MAX), desc='Episode')
    tracker = RewardTracker(EPISODE_MONITOR)
    seed = keras.random.SeedGenerator(seed=SEED)

    states = np.zeros(shape=(MAX_STEPS, DIM_STATES), dtype=np.float32)
    actions = np.zeros(shape=(MAX_STEPS, 1), dtype=np.float32)
    returns = np.zeros(shape=(MAX_STEPS,), dtype=np.float32)

    rewards = []

    for i in pbar:
        total_reward = 0.0
        steps = 0

        states.fill(0)
        actions.fill(0)
        returns.fill(0)
        rewards.clear()

        done = False
        obs, _ = env.reset()

        while not done:
            state = keras.ops.expand_dims(obs, axis=0)
            action = select_action(state, seed).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states[steps] = state
            actions[steps] = action

            rewards.append(reward)

            steps += 1

            obs = next_obs
            total_reward += reward

        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + GAMMA * G
            returns[t] = G

        critic_loss, actor_loss = train(
            states=keras.ops.convert_to_tensor(states),
            actions=keras.ops.convert_to_tensor(actions),
            returns=keras.ops.convert_to_tensor(returns),
            steps=tf.constant(steps)
        )

        tracker.update(total_reward)

        pbar.set_postfix(
            recent_reward=f'{tracker.avg_:.5f}',
            critic_loss=f'{critic_loss:.5f}',
            actor_loss=f'{actor_loss:.5f}'
        )

        if i == 99:
            InvertedPendulumAgent(actor).save('early')

    InvertedPendulumAgent(actor).save('later')

    evaluate('early', InvertedPendulumAgent.load('early'), env, 42)
    evaluate('later', InvertedPendulumAgent.load('later'), env, 42)


if __name__ == '__main__':
    env = gym.make(
        id='InvertedPendulum-v5',
        render_mode='rgb_array'
    )

    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    evaluate('later', InvertedPendulumAgent.load('later'), env, 42)





