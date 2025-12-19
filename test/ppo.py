import kymnasium as kym
from tensorflow.keras import layers, ops, models, optimizers, metrics
import numpy as np
import tensorflow as tf
import os
import pickle
import gymnasium as gym
from concurrent import futures
import shutil
from uuid import uuid4
import logging
import sys
from typing import Literal


def _build_actor():
    input_layer = layers.Input(shape=(3,))

    layer = layers.Dense(units=128, kernel_initializer='he_normal', activation="relu")(input_layer)
    layer = layers.Dense(units=64, kernel_initializer='he_normal', activation="relu")(layer)
    output_mu = layers.Dense(units=1, activation='tanh')(layer)
    output_log_sigma = layers.Dense(units=1, activation='tanh')(layer)

    return models.Model(
        inputs=input_layer,
        outputs=[output_mu, output_log_sigma]
    )


def _build_critic():
    input_layer = layers.Input(shape=(3,))
    layer = layers.Dense(units=128, kernel_initializer='he_normal', activation="relu")(input_layer)
    layer = layers.Dense(units=64, kernel_initializer='he_normal', activation="relu")(layer)
    output_value = layers.Dense(units=1, activation='linear')(layer)

    return models.Model(
        inputs=input_layer,
        outputs=output_value
    )

def _log_prob(
    action, mu, log_sigma
):
    print(action, mu)
    var = ops.square(log_sigma) + 1e-8
    log_prob = -0.5 * ops.log(2 * np.pi) - 0.5 * ops.square(action - mu) / var - log_sigma
    return log_prob


def _train_actor(
        actor, actor_optimizer, clip_ratio,
        buf_state, buf_action, buf_log_prob, buf_advantage
):
    with tf.GradientTape() as tape:
        mu, log_sigma = actor(buf_state)

        log_prob = _log_prob(
            buf_action, mu, log_sigma
        )
        print(log_prob, buf_log_prob)
        ratio = ops.exp(log_prob - buf_log_prob)
        min_advantage = ops.where(
            buf_advantage > 0.0,
            (1 + clip_ratio) * buf_advantage,
            (1 - clip_ratio) * buf_advantage,
        )
        actor_loss = -ops.mean(
            ops.minimum(ratio * buf_advantage, min_advantage)
        )

    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    return actor_loss

@tf.function
def _train_critic(critic, critic_optimizer, buf_state, buf_return):
    with tf.GradientTape() as tape:
        critic_loss = ops.mean(
            ops.square(buf_return - critic(buf_state))
        )
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    return critic_loss

class Agent(kym.Agent):
    PATH_ACTOR = 'actor.keras'
    PATH_CRITIC = 'critic.keras'
    PATH_CONFIG = 'config.pkl'

    def __init__(self, name: str = None, clip_ratio: float = 0.2, iter_train_actor: int = 80,
                 iter_train_critic: int = 80, deterministic: bool = False, frozen: bool = False, gamma: float = 0.99,
                 actor: models.Model = None, actor_optimizer: optimizers.Optimizer = None,
                 actor_loss_tracker: metrics.Mean = None, critic: models.Model = None,
                 critic_optimizer: optimizers.Optimizer = None, critic_loss_tracker: metrics.Mean = None,
                 debug: bool = False):
        self._name = name if name is not None else str(uuid4())
        self._clip_ratio = clip_ratio
        self._iter_train_actor = iter_train_actor
        self._iter_train_critic = iter_train_critic
        self._deterministic = deterministic
        self._frozen = frozen
        self._gamma = gamma

        self._actor = actor if actor is not None else _build_actor()
        self._critic = critic if critic is not None else _build_critic()
        self._actor_optimizer = actor_optimizer if actor_optimizer is not None else optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
        self._actor_loss_tracker = actor_loss_tracker if actor_loss_tracker is not None else metrics.Mean()
        self._critic_optimizer = critic_optimizer if critic_optimizer is not None else optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        self._critic_loss_tracker = critic_loss_tracker if critic_loss_tracker is not None else metrics.Mean()

        self._buf_state = []
        self._buf_value = []
        self._buf_action = []
        self._buf_log_prob = []
        self._buf_reward = []

        formatter = logging.Formatter('[%(levelname)s - %(asctime)s - %(name)s] %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self._logger = logging.getLogger(self._name)
        if debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)


    @property
    def turn_(self):
        return self._turn

    @property
    def name_(self):
        return self._name

    @name_.setter
    def name_(self, name: str):
        self._name = name

    @property
    def frozen_(self):
        return self._frozen

    @frozen_.setter
    def frozen_(self, frozen: bool):
        self._frozen = frozen

    @property
    def actor_(self):
        return self._actor

    @property
    def critic_(self):
        return self._critic

    @property
    def critic_loss_(self):
        return self._critic_loss_tracker.result()

    @property
    def actor_loss_(self):
        return self._actor_loss_tracker.result()

    @classmethod
    def load(cls, path: str) -> 'Agent':
        path_config = os.path.join(path, cls.PATH_CONFIG)
        path_actor = os.path.join(path, cls.PATH_ACTOR)
        path_critic = os.path.join(path, cls.PATH_CRITIC)

        with open(path_config, 'rb') as f:
            config = pickle.loads(f.read())

        return Agent(
            name=config['name'],
            debug=config['debug'],
            clip_ratio=config['clip_ratio'],
            iter_train_actor=config['iter_train_actor'],
            iter_train_critic=config['iter_train_critic'],
            frozen=config['frozen'],
            deterministic=config['deterministic'],
            gamma=config['gamma'],
            actor=models.load_model(path_actor),
            actor_optimizer=optimizers.Adam.from_config(config['actor_optimizer']),
            actor_loss_tracker=metrics.Mean.from_config(config['actor_loss_tracker']),
            critic=models.load_model(path_critic),
            critic_optimizer=optimizers.Adam.from_config(config['critic_optimizer']),
            critic_loss_tracker=metrics.Mean.from_config(config['critic_loss_tracker']),
        )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        path_config = os.path.join(path, self.PATH_CONFIG)
        path_actor = os.path.join(path, self.PATH_ACTOR)
        path_critic = os.path.join(path, self.PATH_CRITIC)

        with open(path_config, 'wb') as f:
            config = {
                'name': self._name,
                'clip_ratio': self._clip_ratio,
                'iter_train_actor': self._iter_train_actor,
                'iter_train_critic': self._iter_train_critic,
                'frozen': self._frozen,
                'deterministic': self._deterministic,
                'gamma': self._gamma,
                'actor_optimizer': self._actor_optimizer.get_config(),
                'critic_optimizer': self._critic_optimizer.get_config(),
                'actor_loss_tracker': self._actor_loss_tracker.get_config(),
                'critic_loss_tracker': self._critic_loss_tracker.get_config(),
            }
            pickle.dump(config, f)
        models.save_model(self._actor, path_actor)
        models.save_model(self._critic, path_critic)

    def _clear_buffer(self):
        self._buf_state.clear()
        self._buf_action.clear()
        self._buf_log_prob.clear()
        self._buf_reward.clear()
        self._buf_value.clear()

    def reset(self):
        self._clear_buffer()
        self._actor_loss_tracker.reset_state()
        self._critic_loss_tracker.reset_state()

    def act(self, observation, info):
        state = ops.expand_dims(observation, axis=0)

        mu, log_sigma = self._actor(state)
        action = mu

        if not self._deterministic:
            action += (np.exp(log_sigma) * np.random.random())

        if not self._frozen:
            log_prob = _log_prob(action, mu, log_sigma)
            self._buf_state.append(state)
            self._buf_action.append(action)
            self._buf_log_prob.append(log_prob)
            self._buf_value.append(self._critic(state))

        return (float(ops.squeeze(ops.clip(action * 2.0, -2.0, 2.0))), )

    def store_reward(self, reward: float):
        self._buf_reward.append(reward)

    def train(self):
        if self._frozen:
            return
        buf_state = ops.cast(ops.concatenate(self._buf_state), dtype='float32')
        buf_action = ops.cast(ops.concatenate(self._buf_action), dtype='float32')
        buf_log_prob = ops.cast(ops.concatenate(self._buf_log_prob), dtype='float32')
        buf_reward = ops.convert_to_tensor(self._buf_reward, dtype='float32')
        buf_value = ops.cast(ops.concatenate(self._buf_value + [ops.convert_to_tensor([[0.0]])]), dtype='float32')

        buf_return = []
        ret = ops.expand_dims(0.0, axis=0)
        for reward in reversed(buf_reward):
            ret = reward + self._gamma * ret
            buf_return.append(ret)
        buf_return = ops.cast(ops.concatenate(buf_return[::-1]), dtype='float32')
        buf_advantage = buf_reward + self._gamma * buf_value[1:] - buf_value[:-1]
        buf_advantage = (buf_advantage - ops.mean(buf_advantage)) / (ops.std(buf_advantage) + 1e-8)

        print('state', buf_state.shape)
        print('action', buf_action.shape)
        print('log_prob', buf_log_prob.shape)
        print('reward', buf_reward.shape)
        print('value', buf_value.shape)
        print('ret', buf_return.shape)
        print('adv', buf_advantage.shape)

        self._logger.debug('--- Training started ---')
        self._logger.debug(f'State: {buf_state}')
        self._logger.debug(f'Action: {buf_action}')
        self._logger.debug(f'Log Prob: {buf_log_prob}')
        self._logger.debug(f'Reward: {buf_reward}')
        self._logger.debug(f'Value: {buf_value}')
        self._logger.debug(f'Return: {buf_return}')
        self._logger.debug(f'Advantage: {buf_advantage}')

        for _ in range(self._iter_train_actor):
            actor_loss = _train_actor(
                self._actor, self._actor_optimizer, self._clip_ratio,
                buf_state, buf_action, buf_log_prob, buf_advantage
            )

            self._actor_loss_tracker.update_state(actor_loss)
        for _ in range(self._iter_train_critic):
            critic_loss = _train_critic(self._critic, self._critic_optimizer, buf_state, buf_return)
            self._critic_loss_tracker.update_state(critic_loss)

        self._clear_buffer()

        self._logger.debug('--- Training completed ---')


if __name__ == '__main__':
    env = gym.make(id='Pendulum-v1', render_mode='rgb_array')
    agent = Agent(frozen=False, debug=False)
    running_rewards = np.zeros(50, dtype='float32')

    for i in range(1000):
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.store_reward(reward)
            total_reward += reward
            done = terminated or truncated
        agent.train()
        running_rewards[i % 50] = total_reward
        print(f'Episode: {i}, Reward: {total_reward:.3f}, Running Reward: {running_rewards.mean():.3f}, Actor Loss: {agent.actor_loss_:.3f}, Critic Loss: {agent.critic_loss_:.3f}')



