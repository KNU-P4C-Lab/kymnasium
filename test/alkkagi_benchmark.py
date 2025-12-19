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
    input_stone = layers.Input(shape=(4,))
    input_mask = layers.Input(shape=(3,))

    layer = layers.Dense(units=128, kernel_initializer='he_normal', activation="relu")(input_stone)
    layer = layers.Dense(units=64, kernel_initializer='he_normal', activation="relu")(layer)

    logits = layers.Dense(units=3, activation='linear')(layer) + (1.0 - input_mask) * -1e8
    output_index = layers.Activation('softmax')(logits)

    output_angle = layers.Dense(units=3, activation='tanh')(layer)

    output_power = layers.Dense(units=3, activation='tanh')(layer)

    return models.Model(
        inputs=[input_stone, input_mask],
        outputs=[output_index, output_angle, output_power]
    )


def _build_critic():
    input_stone = layers.Input(shape=(12,))
    layer = layers.Dense(units=128, kernel_initializer='he_normal', activation="relu")(input_stone)
    layer = layers.Dense(units=64, kernel_initializer='he_normal', activation="relu")(layer)
    output_value = layers.Dense(units=1, activation='linear')(layer)

    return models.Model(
        inputs=[input_stone],
        outputs=[output_value]
    )

def _log_prob(
        index, prob_index,
        angle, angle_mu,
        power, power_mu,
        log_sigma
):
    mask = ops.one_hot(index, 3)
    index_log_prob = ops.log(ops.sum(mask * prob_index, axis=1) + 1e-8)

    angle_mu = ops.sum(mask * angle_mu, axis=1)
    power_mu = ops.sum(mask * power_mu, axis=1)
    var = ops.square(log_sigma) + 1e-8
    angle_log_prob = -0.5 * ops.log(2 * np.pi) - 0.5 * ops.square(angle - angle_mu) / var[0] - log_sigma[0]
    power_log_prob = -0.5 * ops.log(2 * np.pi) - 0.5 * ops.square(power - power_mu) / var[1] - log_sigma[1]
    return index_log_prob + angle_log_prob + power_log_prob


@tf.function
def _train_actor(
        actor, actor_optimizer, clip_ratio, log_sigma,
        buf_state, buf_mask_lived, buf_index, buf_angle, buf_power, buf_log_prob, buf_advantage
):
    with tf.GradientTape() as tape:
        prob_index, angle_mu, power_mu = actor([buf_state, buf_mask_lived])

        log_prob = _log_prob(
            buf_index, prob_index,
            buf_angle, angle_mu,
            buf_power, power_mu, log_sigma
        )

        ratio = ops.exp(log_prob - buf_log_prob)
        min_advantage = ops.where(
            buf_advantage > 0.0,
            (1 + clip_ratio) * buf_advantage,
            (1 - clip_ratio) * buf_advantage,
        )
        actor_loss = -ops.mean(
            ops.minimum(ratio * buf_advantage, min_advantage)
        )

        entropy_discrete = -ops.sum(prob_index * ops.log(prob_index + 1e-8))
        entropy_continuous = ops.sum(log_sigma + 0.5 + 0.5 * ops.log(2 * np.pi))
        entropy_loss = ops.mean(entropy_discrete + entropy_continuous)
        actor_loss -= (0.01 * entropy_loss)

    trianable_variables = actor.trainable_variables + [log_sigma]
    actor_grads = tape.gradient(actor_loss, trianable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, trianable_variables))

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

    def __init__(self, mode: Literal['random', 'heuristic', 'rl'], turn: int, name: str = None, clip_ratio: float = 0.2, iter_train_actor: int = 80,
                 iter_train_critic: int = 80, deterministic: bool = False, frozen: bool = False, gamma: float = 0.99,
                 actor: models.Model = None, actor_optimizer: optimizers.Optimizer = None,
                 actor_loss_tracker: metrics.Mean = None, critic: models.Model = None,
                 critic_optimizer: optimizers.Optimizer = None, critic_loss_tracker: metrics.Mean = None,
                 win_rate_tracker: metrics.Mean = None, debug: bool = False):
        self._mode = mode
        self._turn = turn
        self._name = name if name is not None else str(uuid4())
        self._clip_ratio = clip_ratio
        self._iter_train_actor = iter_train_actor
        self._iter_train_critic = iter_train_critic
        self._deterministic = deterministic
        self._frozen = frozen
        self._gamma = gamma
        self._debug = debug

        self._actor = actor if actor is not None else _build_actor()
        self._critic = critic if critic is not None else _build_critic()
        self._actor_optimizer = actor_optimizer if actor_optimizer is not None else optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
        self._actor_loss_tracker = actor_loss_tracker if actor_loss_tracker is not None else metrics.Mean()
        self._critic_optimizer = critic_optimizer if critic_optimizer is not None else optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        self._critic_loss_tracker = critic_loss_tracker if critic_loss_tracker is not None else metrics.Mean()
        self._win_rate_tracker = win_rate_tracker if win_rate_tracker is not None else metrics.Mean()

        self._buf_state = []
        self._buf_mask_lived = []
        self._buf_value = []
        self._buf_index = []
        self._buf_angle = []
        self._buf_power = []
        self._buf_log_prob = []
        self._buf_reward = []

        self._log_sigma = tf.Variable(np.zeros(shape=(2, ), dtype='float32')  - 0.5, trainable=True)

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
    def mode_(self):
        return self._mode

    @mode_.setter
    def mode_(self, mode: Literal['random', 'heuristic', 'rl']):
        self._mode = mode

    @property
    def turn_(self):
        return self._turn

    @turn_.setter
    def turn_(self, turn: int):
        self._turn = turn

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

    @property
    def win_rate_(self):
        return self._win_rate_tracker.result()

    @classmethod
    def load(cls, path: str) -> 'Agent':
        path_config = os.path.join(path, cls.PATH_CONFIG)
        path_actor = os.path.join(path, cls.PATH_ACTOR)
        path_critic = os.path.join(path, cls.PATH_CRITIC)

        with open(path_config, 'rb') as f:
            config = pickle.loads(f.read())

        return Agent(
            turn=config['turn'],
            name=config['name'],
            mode=config['mode'],
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
            win_rate_tracker=metrics.Mean.from_config(config['win_rate_tracker'])
        )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        path_config = os.path.join(path, self.PATH_CONFIG)
        path_actor = os.path.join(path, self.PATH_ACTOR)
        path_critic = os.path.join(path, self.PATH_CRITIC)

        with open(path_config, 'wb') as f:
            config = {
                'turn': self._turn,
                'name': self._name,
                'mode': self._mode,
                'debug': self._debug,
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
                'win_rate_tracker': self._win_rate_tracker.get_config()
            }
            pickle.dump(config, f)
        models.save_model(self._actor, path_actor)
        models.save_model(self._critic, path_critic)

    def _preprocess(self, observation: dict):
        black, white = observation['black'], observation['white']
        black_mask, white_mask = black[:, [2]] > 0, white[:, [2]] > 0
        black, white = black[:, :2] * black_mask / 300.0 - 1.0, white[:, :2] * white_mask / 300.0 - 1.0
        state = np.concat([np.ravel(black), np.ravel(white)])

        return state, np.squeeze(black_mask if self._turn == 0 else white_mask).astype(int)

    def _clear_buffer(self):
        self._buf_state.clear()
        self._buf_mask_lived.clear()
        self._buf_index.clear()
        self._buf_angle.clear()
        self._buf_power.clear()
        self._buf_log_prob.clear()
        self._buf_reward.clear()
        self._buf_value.clear()

    def reset(self):
        self._clear_buffer()
        self._actor_loss_tracker.reset_state()
        self._critic_loss_tracker.reset_state()
        self._win_rate_tracker.reset_state()

    def update_win_rate(self, win: bool):
        self._win_rate_tracker.update_state(1.0 if win else 0.0)

    def _act_random(self, observation, info):
        return {
            'turn': self._turn,
            'angle': float(np.random.uniform(-180, 180)),
            'power': float(np.random.uniform(1, 2500)),
            'index': float(np.random.choice(3))
        }

    def _act_heuristic(self, observation, info):
        stone_self = observation['black'] if self._turn == 0 else observation['white']
        stone_opponent = observation['white'] if self._turn == 0 else observation['black']
        index_self = np.random.choice(np.flatnonzero(stone_self[:, 2] > 0))
        index_opponent = np.random.choice(np.flatnonzero(stone_opponent[:, 2] > 0) )
        dx, dy = stone_opponent[index_opponent, :2] - stone_self[index_self, :2]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        nx, ny = dx / (dist + 1e-8), dy / (dist + 1e-8)
        radian = np.arctan2(ny, nx)  if nx < 0 else np.arctan2(ny, nx)
        angle = np.rad2deg(radian)

        return {
            'turn': self._turn,
            'angle': float(np.clip(angle, -180.0, 180.0)),
            'power': float(np.random.uniform(1000, 2500)),
            'index': int(index_self)
        }

    def _act_rl(self, observation, info):
        state, mask_lived = self._preprocess(observation)
        prob_index, angle_mu, power_mu = self._actor([
            ops.expand_dims(state, axis=0), ops.expand_dims(mask_lived, axis=0)
        ])

        if self._deterministic:
            index = np.random.choice(np.flatnonzero(np.squeeze(prob_index) == np.max(prob_index)))
        else:
            index = np.random.choice(3, p=np.squeeze(prob_index))

        chosen_mask = ops.one_hot(index, 3)

        angle, power = ops.sum(angle_mu * chosen_mask), ops.sum(power_mu * chosen_mask)

        if not self._deterministic:
            angle_log_sigma, power_log_sigma = self._log_sigma
            angle += (np.exp(angle_log_sigma) * np.random.random())
            power += (np.exp(power_log_sigma) * np.random.random())

        if self._turn == 1:
            angle = -angle

        if not self._frozen:
            log_prob = _log_prob(index, prob_index, angle, angle_mu, power, power_mu, self._log_sigma)
            self._buf_state.append(state)
            self._buf_mask_lived.append(mask_lived)
            self._buf_index.append(index)
            self._buf_angle.append(angle)
            self._buf_power.append(power)
            self._buf_log_prob.append(ops.squeeze(log_prob))
            self._buf_value.append(ops.squeeze(self._critic(ops.expand_dims(state, axis=0))))

        return {
            'turn': int(self._turn),
            'angle': float(ops.squeeze(ops.clip(angle * 180.0, -180.0, 180.0))),
            'power': float(ops.squeeze(ops.clip((power + 1.0) * 1000 + 500, 1.0, 2500.0))),
            'index': int(index)
        }

    def act(self, observation, info):
        if self._turn != observation['turn']:
            return None
        if self._mode == 'random':
            return self._act_random(observation, info)
        elif self._mode == 'heuristic':
            return self._act_heuristic(observation, info)
        else:
            return self._act_rl(observation, info)

    def store_reward(self, reward: float):
        self._buf_reward.append(reward)

    def train(self):
        if self._frozen or self._mode != 'rl':
            return

        buf_state = ops.convert_to_tensor(self._buf_state, dtype='float32')
        buf_mask_lived = ops.convert_to_tensor(self._buf_mask_lived, dtype='int32')
        buf_index = ops.convert_to_tensor(self._buf_index, dtype='int32')
        buf_angle = ops.convert_to_tensor(self._buf_angle, dtype='float32')
        buf_power = ops.convert_to_tensor(self._buf_power, dtype='float32')
        buf_log_prob = ops.convert_to_tensor(self._buf_log_prob, dtype='float32')
        buf_reward = ops.convert_to_tensor(self._buf_reward, dtype='float32')
        buf_value = ops.convert_to_tensor(self._buf_value + [0.0], dtype='float32')

        buf_return = []
        ret = 0.0
        for reward in reversed(buf_reward):
            ret = reward + self._gamma * ret
            buf_return.append(ret)
        buf_return = ops.convert_to_tensor(buf_return[::-1], dtype='float32')
        buf_advantage = buf_reward + self._gamma * buf_value[1:] - buf_value[:-1]
        buf_advantage = (buf_advantage - ops.mean(buf_advantage)) / (ops.std(buf_advantage) + 1e-8)

        self._logger.debug('--- Training started ---')
        self._logger.debug(f'State: {buf_state}')
        self._logger.debug(f'Mask: {buf_mask_lived}')
        self._logger.debug(f'Index: {buf_index}')
        self._logger.debug(f'Angle: {buf_angle}')
        self._logger.debug(f'Power: {buf_power}')
        self._logger.debug(f'Log Prob: {buf_log_prob}')
        self._logger.debug(f'Reward: {buf_reward}')
        self._logger.debug(f'Value: {buf_value}')
        self._logger.debug(f'Return: {buf_return}')
        self._logger.debug(f'Advantage: {buf_advantage}')

        for _ in range(self._iter_train_actor):
            actor_loss = _train_actor(
                self._actor, self._actor_optimizer, self._clip_ratio, self._log_sigma,
                buf_state, buf_mask_lived, buf_index, buf_angle, buf_power, buf_log_prob, buf_advantage
            )
            self._actor_loss_tracker.update_state(actor_loss)
        for _ in range(self._iter_train_critic):
            critic_loss = _train_critic(self._critic, self._critic_optimizer, buf_state, buf_return)
            self._critic_loss_tracker.update_state(critic_loss)

        self._clear_buffer()

        self._logger.debug('--- Training completed ---')


def demo2(black_agent: kym.Agent, white_agent: kym.Agent):
    env = gym.make('kymnasium/AlKkaGi-3x3-v0', render_mode='human', obs_type='custom')
    obs, info = env.reset()
    done = False

    while not done:
        if obs['turn'] == 0:
            action = black_agent.act(obs, info)
        else:
            action = white_agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


def test():
    print('asdf')


def run(
        path_pool: str,
        path_winner: str,
        max_workers: int,
        pool_size: int,
        max_round: int,
        track_round: int,
        win_threshold: float,
        max_generation: int
):
    os.makedirs(path_pool, exist_ok=True)
    os.makedirs(path_winner, exist_ok=True)

    n_init_agents = len(os.listdir(path_pool))

    while n_init_agents < pool_size:
        agent = Agent(
            turn=0, frozen=False, deterministic=False
        )
        agent.save(os.path.join(path_pool, agent.name_))
        n_init_agents += 1

    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for gen in range(max_generation):
            tasks = []
            pools = os.listdir(path_pool)

            for _ in range(pool_size):
                train_agent = Agent.load(os.path.join(path_pool, pools[np.random.choice(len(pools))]))
                frozen_agent = Agent.load(os.path.join(path_pool, pools[np.random.choice(len(pools))]))
                task = executor.submit(fight, train_agent, frozen_agent, track_round, win_threshold, max_round)
                tasks.append(task)

            for task in futures.as_completed(tasks):
                agent = task.result()
                agent.name_ = str(uuid4())
                agent.save(os.path.join(path_winner, agent.name_))

            for f in pools:
                shutil.rmtree(os.path.join(path_pool, f), ignore_errors=True)

            for f in os.listdir(path_winner):
                shutil.move(os.path.join(path_winner,f), os.path.join(path_pool, f))


def fight(
        train_agent: Agent,
        frozen_agent: Agent,
        track_round: int,
        win_threshold: float,
        max_round: int,
):
    env = gym.make('kymnasium/AlKkaGi-3x3-v0', render_mode='rgb_array', obs_type='custom', bgm=False)

    print(f'{train_agent.name_} vs. {frozen_agent.name_}: Started')

    train_agent.reset()
    frozen_agent.reset()

    history_win = np.zeros(shape=(track_round,), dtype='bool')
    history_reward = np.zeros(shape=(track_round,), dtype='float32')
    for i in range(max_round):
        obs, info = env.reset()
        done = False

        #train_agent.turn_ = i % 2
        #frozen_agent.turn_ = (i + 1) % 2
        train_agent.turn_ = 0
        frozen_agent.turn_ = 1
        steps = 0
        reward = 0.0
        total_reward = 0.0
        while not done:
            if obs['turn'] == train_agent.turn_:
                action = train_agent.act(obs, info)
            else:
                action = frozen_agent.act(obs, info)

            next_obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated or steps >= 30
            cur_self = obs['black'] if train_agent.turn_ == 0 else obs['white']
            cur_opponent = obs['white'] if train_agent.turn_ == 0 else obs['black']

            cur_self = np.sum(cur_self[:, 2] > 0)
            cur_opponent = np.sum(cur_opponent[:, 2] > 0)

            next_self = next_obs['black'] if train_agent.turn_ == 0 else next_obs['white']
            next_opponent = next_obs['white'] if train_agent.turn_ == 0 else next_obs['black']

            next_self = np.sum(next_self[:, 2] > 0)
            next_opponent = np.sum(next_opponent[:, 2] > 0)

            if obs['turn'] == train_agent.turn_:
                suicide = (cur_self - next_self) * -0.33
                knockout = (cur_opponent - next_opponent) * 0.33
                reward += (suicide + knockout)

            if done:
                reward += 1.0 if next_self > next_opponent else -1.0

            if done or obs['turn'] != train_agent.turn_:
                train_agent.store_reward(reward)
                total_reward += reward
                reward = 0.0
            '''
            if done:
                next_self = next_obs['black'] if train_agent.turn_ == 0 else next_obs['white']
                next_opponent = next_obs['white'] if train_agent.turn_ == 0 else next_obs['black']

                next_self = np.sum(next_self[:, 2] > 0)
                next_opponent = np.sum(next_opponent[:, 2] > 0)
                reward = 1.0 if next_self > next_opponent else -1.0
                train_agent.store_reward(reward)
                total_reward += reward
            elif obs['turn'] != train_agent.turn_:
                train_agent.store_reward(0.0)
            '''
            obs = next_obs

        black_count = np.sum(obs['black'][:, 2] > 0)
        white_count = np.sum(obs['white'][:, 2] > 0)

        if black_count > white_count:
            win = 0
        elif black_count < white_count:
            win = 1
        else:
            win = frozen_agent.turn_

        train_agent.train()
        print(f'Actor Loss = {train_agent.actor_loss_:.5f} / Critic Loss = {train_agent.critic_loss_:.5f}')
        train_agent.update_win_rate(win == train_agent.turn_)
        frozen_agent.update_win_rate(win == frozen_agent.turn_)

        history_win[i % track_round] = win == train_agent.turn_
        history_reward[i % track_round] = total_reward
        if i >= track_round and np.mean(history_win) >= win_threshold:
            break

        if i % 50 == 0 and i > 0:
            print(f'{train_agent.name_} vs. {frozen_agent.name_}: Round = {i}, Win Rate = {np.mean(history_win):.4f} / Reward = {np.mean(history_reward):.4f}')

    print(f'{train_agent.name_} vs. {frozen_agent.name_}: Completed; Win Rate = {np.mean(history_win):.4f}')

    return train_agent

if __name__ == '__main__':
    agent = fight(Agent.load('test'), Agent(turn=1, mode='heuristic'), 50, 0.6, 5000)
    agent.save('test')
    #demo(PPOAgent.load('test'), PPOAgent(turn=1, frozen=False, deterministic=False))

