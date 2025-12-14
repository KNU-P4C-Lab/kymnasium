import kymnasium as kym
from tensorflow.keras import layers, ops, models, optimizers, metrics
import numpy as np
import tensorflow as tf
import os
import pickle
import gymnasium as gym
from concurrent import futures

class Agent(kym.Agent):
    N_BUFFER = 200
    PATH_ACTOR = 'actor.keras'
    PATH_CRITIC = 'critic.keras'
    PATH_CONFIG = 'config.pkl'

    def __init__(
            self,
            turn: int,
            clip_ratio: float=0.2,
            iter_train_actor: int=80,
            iter_train_critic: int=80,
            training: bool = False,
            gamma: float = 0.99,
            rating: float = 1500.0,
            actor: models.Model = None,
            actor_optimizer: optimizers.Optimizer = None,
            actor_loss_tracker: metrics.Mean = None,
            critic: models.Model = None,
            critic_optimizer: optimizers.Optimizer = None,
            critic_loss_tracker: metrics.Mean = None,
    ):
        self._turn = turn
        self._clip_ratio = clip_ratio
        self._iter_train_actor = iter_train_actor
        self._iter_train_critic = iter_train_critic
        self._training = training
        self._gamma = gamma
        self._rating = rating

        self._actor = actor if actor is not None else self._build_actor()
        self._critic = critic if critic is not None else self._build_critic()
        self._actor_optimizer = actor_optimizer if actor_optimizer is not None else optimizers.Adam(learning_rate=3e-4)
        self._actor_loss_tracker = actor_loss_tracker if actor_loss_tracker is not None else metrics.Mean()
        self._critic_optimizer = critic_optimizer if critic_optimizer is not None else optimizers.Adam(learning_rate=1e-3)
        self._critic_loss_tracker = critic_loss_tracker if critic_loss_tracker is not None else metrics.Mean()

        self._buf_black = np.zeros(shape=(self.N_BUFFER, 6), dtype='float32')
        self._buf_white = np.zeros(shape=(self.N_BUFFER, 6), dtype='float32')
        self._buf_value = np.zeros(shape=(self.N_BUFFER, ), dtype='float32')
        self._buf_index = np.zeros(shape=(self.N_BUFFER, ), dtype='float32')
        self._buf_angle = np.zeros(shape=(self.N_BUFFER, ), dtype='float32')
        self._buf_power = np.zeros(shape=(self.N_BUFFER, ), dtype='float32')
        self._buf_log_prob = np.zeros(shape=(self.N_BUFFER, ), dtype='float32')
        self._counter = 0

    @property
    def training_(self):
        return self._training

    @training_.setter
    def training_(self, training: bool):
        self._training = training

    @property
    def rating_(self):
        return self._rating

    @property
    def turn_(self):
        return self._turn

    @turn_.setter
    def turn_(self, turn: int):
        self._turn = turn

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

    def train(self, is_win: bool):
        if not self._training:
            self._counter = 0
            return

        ret = 1.0 if is_win else -1.0
        self._buf_value[self._counter + 1] = ret

        buf_return = []
        for _ in range(self._counter):
            ret *= self._gamma
            buf_return.append(ret)
        buf_return = np.array(buf_return[::-1], dtype='float32')
        buf_advantage = self._gamma * self._buf_value[1:self._counter + 1] - self._buf_value[:self._counter]
        buf_black = self._buf_black[:self._counter]
        buf_white = self._buf_white[:self._counter]
        buf_index = self._buf_index[:self._counter]
        buf_power = self._buf_power[:self._counter]
        buf_angle = self._buf_angle[:self._counter]
        buf_action_probs = self._buf_log_prob[:self._counter]

        for _ in range(self._iter_train_actor):
            with tf.GradientTape() as tape:
                idx_probs, angle_mu, angle_log_sigma, power_mu, power_log_sigma = self._actor([buf_black, buf_white])
                angle_mu, angle_log_sigma = ops.squeeze(angle_mu), ops.squeeze(angle_log_sigma)
                power_mu, power_log_sigma = ops.squeeze(power_mu), ops.squeeze(power_log_sigma)
                log_probs = self._log_prob(buf_index, idx_probs, buf_angle, angle_mu, angle_log_sigma, buf_power, power_mu, power_log_sigma)
                ratio = ops.exp(log_probs - buf_action_probs)
                min_advantage = ops.where(
                    buf_advantage > 0.0,
                    (1 + self._clip_ratio) * buf_advantage,
                    (1 - self._clip_ratio) * buf_advantage,
                )
                actor_loss = -ops.mean(
                    ops.minimum(ratio * buf_advantage, min_advantage)
                )
            actor_grads = tape.gradient(actor_loss, self._actor.trainable_variables)
            self._actor_optimizer.apply_gradients(zip(actor_grads, self._actor.trainable_variables))

        for _ in range(self._iter_train_critic):
            with tf.GradientTape() as tape:
                critic_loss = ops.mean(
                    metrics.huber(buf_return, self._critic([buf_black, buf_white]))
                )
            critic_grads = tape.gradient(critic_loss, self._critic.trainable_variables)
            self._critic_optimizer.apply_gradients(zip(critic_grads, self._critic.trainable_variables))

        self._actor_loss_tracker.update_state(actor_loss)
        self._critic_loss_tracker.update_state(critic_loss)

        self._counter = 0

    def update_rating(self, is_win: bool, opponent_rating: float):
        expected_score = 1.0 / (1 + 10 ** ((opponent_rating - self._rating) / 400.0))

        if self._rating > 2400:
            factor = 16
        elif 2100 < self._rating <= 2400:
            factor = 24
        else:
            factor = 32

        self._rating += factor * (1.0 if is_win else 0.0 - expected_score)

    @classmethod
    def load(cls, path: str) -> 'Agent':
        path_config = os.path.join(path, cls.PATH_CONFIG)
        path_actor = os.path.join(path, cls.PATH_ACTOR)
        path_critic = os.path.join(path, cls.PATH_CRITIC)

        with open(path_config, 'rb') as f:
            config = pickle.loads(f.read())

        actor = models.load_model(path_actor)
        critic = models.load_model(path_critic)

        return Agent(
            turn=config['turn'],
            clip_ratio=config['clip_ratio'],
            iter_train_actor=config['iter_train_actor'],
            iter_train_critic=config['iter_train_critic'],
            training=config['training'],
            gamma=config['gamma'],
            rating=config['rating'],
            actor=actor,
            actor_optimizer=optimizers.Adam.from_config(config['actor_optimizer']),
            actor_loss_tracker=metrics.Metric.from_config(config['actor_loss_tracker']),
            critic=critic,
            critic_optimizer=optimizers.Adam.from_config(config['critic_optimizer']),
            critic_loss_tracker=metrics.Metric.from_config(config['critic_loss_tracker'])
        )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        path_config = os.path.join(path, self.PATH_CONFIG)
        path_actor = os.path.join(path, self.PATH_ACTOR)
        path_critic = os.path.join(path, self.PATH_CRITIC)

        with open(path_config, 'wb') as f:
            config = {
                'turn': self._turn,
                'clip_ratio': self._clip_ratio,
                'iter_train_actor': self._iter_train_actor,
                'iter_train_critic': self._iter_train_critic,
                'training': self._training,
                'gamma': self._gamma,
                'rating': self._rating,
                'actor_optimizer': self._actor_optimizer.get_config(),
                'critic_optimizer': self._critic_optimizer.get_config(),
                'actor_loss_tracker': self._actor_loss_tracker.get_config(),
                'critic_loss_tracker': self._critic_loss_tracker.get_config()
            }
            pickle.dump(config, f)
        models.save_model(self._actor, path_actor)
        models.save_model(self._critic, path_critic)

    @classmethod
    def _preprocess(cls, stones: np.ndarray):
        stones = np.array(stones)

        for i in range(2):
            stones[:, i] = np.where(stones[:, 2] < 1, 0, stones[:, i])
        stones = stones / 300.0 - 1.0
        stones = np.ravel(stones[:, :2])
        return stones

    @classmethod
    def _postprocess(cls, angle, power):
        angle *= 180.0
        power *= 2500.0
        angle = np.clip(angle, -180.0, 180.0)
        power = np.clip(power, 1.0, 2500.0)
        return angle, power

    @classmethod
    def _build_actor(cls):
        input_black = layers.Input(shape=(6,))
        input_white = layers.Input(shape=(6,))
        concat = layers.Concatenate()([input_black, input_white])
        layer = layers.Dense(units=512, kernel_initializer='he_normal', activation="relu")(concat)
        layer = layers.Dense(units=256, kernel_initializer='he_normal', activation="relu")(layer)
        layer = layers.Dense(units=128, kernel_initializer='he_normal', activation="relu")(layer)
        output_index = layers.Dense(units=3, activation='softmax')(layer)
        output_angle_mu = layers.Dense(units=1, activation='tanh')(layer)
        output_angle_log_sigma = layers.Dense(units=1, activation='tanh')(layer) * 0.5
        output_power_mu = layers.Dense(units=1, activation='sigmoid')(layer)
        output_power_log_sigma = layers.Dense(units=1, activation='tanh')(layer) * 0.5

        return models.Model(
            inputs=[input_black, input_white],
            outputs=[output_index, output_angle_mu, output_angle_log_sigma, output_power_mu, output_power_log_sigma]
        )

    @classmethod
    def _build_critic(cls):
        input_black = layers.Input(shape=(6,))
        input_white = layers.Input(shape=(6,))
        concat = layers.Concatenate()([input_black, input_white])
        layer = layers.Dense(units=512, kernel_initializer='he_normal', activation="relu")(concat)
        layer = layers.Dense(units=256, kernel_initializer='he_normal', activation="relu")(layer)
        layer = layers.Dense(units=128, kernel_initializer='he_normal', activation="relu")(layer)
        output = layers.Dense(units=1, activation='linear')(layer)
        return models.Model(
            inputs=[input_black, input_white],
            outputs=[output]
        )

    @classmethod
    def _log_prob(cls, idx, idx_probs, angle, angle_mu, angle_log_sigma, power, power_mu, power_log_sigma):
        idx_log_prob = ops.log(ops.sum(ops.one_hot(idx, 3) * idx_probs, axis=1))
        angle_log_prob = -angle_log_sigma - 0.5 * ops.log(2 * np.pi) - 0.5 * ops.square( (angle - angle_mu) / (ops.exp(angle_log_sigma) + 1e-5) )
        power_log_prob = -power_log_sigma - 0.5 * ops.log(2 * np.pi) - 0.5 * ops.square( (power - power_mu) / (ops.exp(power_log_sigma) + 1e-5) )

        return idx_log_prob + angle_log_prob + power_log_prob

    def act(self, observation, info):
        turn, black, white = observation['turn'], observation['black'], observation['white']
        if self._turn != turn:
            return None

        black, white = self._preprocess(black), self._preprocess(white)
        inputs = [ops.expand_dims(black, axis=0), ops.expand_dims(white, axis=0)]
        idx_probs, angle_mu, angle_log_sigma, power_mu, power_log_sigma = self._actor(inputs)

        if self._training:
            idx = np.random.choice(3, p=np.squeeze(idx_probs))
            angle = angle_mu + ops.exp(angle_log_sigma) * np.random.normal()
            power = power_mu + ops.exp(power_log_sigma) * np.random.normal()
            log_prob = self._log_prob(idx, idx_probs, angle, angle_mu, angle_log_sigma, power, power_mu, power_log_sigma)
            angle, power, log_prob = ops.squeeze(angle), ops.squeeze(power), ops.squeeze(log_prob)
            value = ops.squeeze(self._critic(inputs))

            self._buf_black[self._counter] = black
            self._buf_white[self._counter] = white
            self._buf_index[self._counter] = idx
            self._buf_angle[self._counter] = angle
            self._buf_power[self._counter] = power
            self._buf_log_prob[self._counter] = log_prob
            self._buf_value[self._counter] = value
        else:
            idx = np.random.choice(np.squeeze(idx_probs) == np.max(idx_probs))
            angle, power = ops.squeeze(power_mu), ops.squeeze(power_mu)

        angle, power = self._postprocess(angle, power)
        if self._turn == 1:
            angle = -angle

        self._counter += 1

        return {
            'turn': self._turn,
            'angle': angle,
            'power': power,
            'index': int(idx)
        }

def train_test():
    black_agent = Agent(turn=0, training=True)
    white_agent = Agent(turn=1, training=False)

    env = gym.make('kymnasium/AlKkaGi-3x3-v0', render_mode='rgb_array', obs_type='custom')

    for _ in range(1000):
        obs, info = env.reset()
        done = False

        while not done:
            if obs['turn'] == 0:
                action = black_agent.act(obs, info)
            else:
                action = white_agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        black_win = np.all(obs['white'][:, 2] < 1)
        black_agent.train(black_win)
        white_agent.train(not black_win)
        black_agent.update_rating(black_win, white_agent.rating_)
        white_agent.update_rating(not black_win, black_agent.rating_)
        print(f'[Black]: Win = {black_win} / Actor = {black_agent.actor_loss_:.5f} / Critic = {black_agent.critic_loss_:.5f}')
        print(f'Black rating = {black_agent.rating_} / White rating = {white_agent.rating_}')


def fight(n_round: int, black_agent: Agent, white_agent: Agent):
    env = gym.make('kymnasium/AlKkaGi-3x3-v0', render_mode='rgb_array', obs_type='custom')

    for _ in range(n_round):
        obs, info = env.reset()
        done = False

        while not done:
            if obs['turn'] == 0:
                action = black_agent.act(obs, info)
            else:
                action = white_agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        black_win = np.all(obs['white'][:, 2] < 1)
        black_agent.train(black_win)
        white_agent.train(not black_win)
        black_agent.update_rating(black_win, white_agent.rating_)
        white_agent.update_rating(not black_win, black_agent.rating_)

    return black_agent, white_agent


def run(n_round: int, pool: int, max_workers: int = None):
    agents = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        init_agents = [Agent(turn=0, training=True) for _ in range(pool)]
        tasks = [
            executor.submit(fight, n_round, agent, Agent(turn=1, training=False))
            for agent in init_agents
        ]
        for task in futures.as_completed(tasks):
            agent, _ = task.result()
            agents.append(agent)




if __name__ == '__main__':
    train_test()