from tensorflow import keras
import tensorflow as tf
import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm
from tensorflow_probability import distributions as tfd
import os
import kymnasium as kym
from typing import Any, Dict




class InvertedPendulumAgent(kym.Agent):
    def __init__(
            self,
            model: keras.models.Model,
    ):
        self.model = model

    def save(self, path: str):
        keras.models.save_model(
            model=self.model,
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
        mu, _ = self.model(state)
        mu = keras.ops.squeeze(mu)
        action = keras.ops.clip(mu, -3.0, 3.0)
        action = action.numpy()

        return (action, )

def train():
    env = gym.make(
        id='InvertedPendulum-v5',
        render_mode='rgb_array'
    )

    input = keras.layers.Input(
        shape=env.observation_space.shape,
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

    mu = keras.layers.Dense(
        units=1,
        activation=keras.activations.tanh,
        kernel_initializer=keras.initializers.GlorotNormal(seed=42)
    )(layer) * 2.0

    log_sigma = keras.layers.Dense(
        units=1,
        activation=keras.activations.tanh,
        kernel_initializer=keras.initializers.GlorotNormal(seed=42)
    )(layer) * 3.0 - 2.0

    actor = keras.models.Model(
        inputs=input,
        outputs=[mu, log_sigma]
    )

    critic = keras.models.Sequential([
        keras.layers.Input(
            shape=env.observation_space.shape
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

    actor_optimizer = keras.optimizers.Adam(
        learning_rate=0.00003,
        clipnorm=1.0
    )

    critic_optimizer = keras.optimizers.Adam(
        learning_rate=0.0001,
        clipnorm=1.0
    )

    EPISODES_MAX = 1000
    GAMMA = 0.99

    reward_per_episode = np.zeros(30)
    history = []
    critic_objective = keras.losses.Huber()

    pbar = tqdm(range(EPISODES_MAX), desc='Episode')

    for episode in pbar:
        history.clear()
        episode_reward = 0

        done = False
        obs, _ = env.reset()

        while not done:
            state = keras.ops.expand_dims(obs, axis=0)

            mu, log_sigma = actor(state)
            mu, log_sigma = keras.ops.squeeze(mu), keras.ops.squeeze(log_sigma)
            sigma = keras.ops.exp(log_sigma)
            action = tfd.Normal(loc=mu, scale=sigma).sample().numpy()
            action = np.clip(action, -2.0, 2.0)

            value = np.squeeze(
                critic(state)
            )
            obs, reward, terminated, truncated, _ = env.step((action,))

            episode_reward += reward
            done = terminated or truncated

            history.append((state, action, reward, value))

        batch_state = [exp[0] for exp in history]
        batch_action = [exp[1] for exp in history]
        batch_reward = [exp[2] for exp in history]
        batch_value = [exp[3] for exp in history]

        batch_return = []
        G = 0.0
        for reward in reversed(batch_reward):
            G = reward + GAMMA * G
            batch_return.insert(0, G)

        batch_state = keras.ops.concatenate(batch_state)
        batch_return = keras.ops.convert_to_tensor(batch_return, dtype='float32')
        batch_value = keras.ops.convert_to_tensor(batch_value)

        for _ in range(10):
            with tf.GradientTape() as tape:
                value = keras.ops.squeeze(
                    critic(batch_state)
                )
                critic_loss = critic_objective(batch_return, value)
            critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        with tf.GradientTape() as tape:
            # 위에서 봤던대로 선택한 행동에 대한 로그 확률을 출력한다.
            mu, log_sigma = actor(batch_state)
            mu, log_sigma = keras.ops.squeeze(mu), keras.ops.squeeze(log_sigma)
            sigma = keras.ops.exp(log_sigma)
            log_prob = tfd.Normal(loc=mu, scale=sigma).log_prob(batch_action)

            target = batch_return - batch_value
            target = (target - keras.ops.mean(target)) / (keras.ops.std(target) + 1e-8)

            actor_loss = log_prob * target * (GAMMA ** np.arange(len(history)))
            actor_loss = -keras.ops.sum(actor_loss)
        actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))

        reward_per_episode[episode % len(reward_per_episode)] = episode_reward
        running_reward = np.mean(reward_per_episode)

        pbar.set_postfix(
            actor_loss=f'{actor_loss:.5f}',
            critic_loss=f'{critic_loss:.5f}',
            reward=f'{episode_reward:.5f}',
            running_reward=f'{running_reward:.5f}',
        )

        if (episode + 1) % 200 == 0:
            os.makedirs('./files/pend', exist_ok=True)
            keras.models.save_model(actor, f'./files/pend/{episode}.keras')


def record(env, model, path, prefix):
    agent = InvertedPendulumAgent(model)
    recoder = gym.wrappers.RecordVideo(
        env=env,
        video_folder=path,
        name_prefix=prefix
    )
    done = False
    obs, info = recoder.reset()

    while not done:
        action = agent.act(obs, info)
        obs, _, terminated, truncated, info = recoder.step(action)
        done = terminated or truncated

    recoder.close()



if __name__ == '__main__':
    # train()
    env = gym.make(id='InvertedPendulum-v5', render_mode='rgb_array')

    #model = keras.models.load_model('./files/pend/199.keras')
    #record(env, model, './files/pend', 'early')

    model = keras.models.load_model('./files/pend/999.keras')
    record(env, model, './files/pend', 'later')