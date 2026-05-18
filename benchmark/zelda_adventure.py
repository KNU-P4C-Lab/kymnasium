import kymnasium as kym
import gymnasium as gym
import numpy as np
import pickle
import os
from collections import UserDict
from tqdm.auto import tqdm
import json
from dataclasses import dataclass

class VisitsCounter(UserDict):
    def __init__(self, max_size: int):
        super().__init__()
        self._max_size = max_size

    def __setitem__(self, key: tuple, value: float):
        if len(self.data) >= self._max_size:
            del self.data[next(iter(self.data))]
        self.data[key] = value

    def __getitem__(self, key: tuple):
        if key not in self.data:
            return 0
        return self.data[key]


class ActionValue(UserDict):
    def __init__(
            self,
            n_actions: int,
            seed: int = None,
            scale: float = 1.0,
            offset: float = 0.0,
    ):
        super().__init__()
        self._n_actions = n_actions
        self._random = np.random.default_rng(seed)
        self._scale = scale
        self._offset = offset

    def __getitem__(self, key: tuple):
        if key not in self.data:
            init_value = self._random.normal(size=(self._n_actions,)) * self._scale + self._offset
            self.data[key] = init_value
        return self.data[key]



class ResultTracker:
    def __init__(self, n_monitor: int = 100):
        self._n_monitor = n_monitor
        self._results = []
        self._episode = 0

    def update(self, reward, death, timeout, cleared):
        if len(self._results) > self._n_monitor:
            del self._results[:1]

        self._results.append((reward, death, timeout, cleared))
        self._episode += 1

    @property
    def n_episode_(self):
        return self._episode

    @property
    def mean_rewards_(self):
        return np.mean([r[0] for r in self._results])

    @property
    def best_rewards_(self):
        return np.max([r[0] for r in self._results])

    @property
    def death_ratio_(self):
        return np.mean([r[1] for r in self._results])

    @property
    def timeout_ratio_(self):
        return np.mean([r[2] for r in self._results])

    @property
    def cleared_ratio_(self):
        return np.mean([r[3] for r in self._results])


class ZeldaQAgent(kym.Agent):
    ACTIONS = [0, 1, 2, 3, 4, 5, 6]
    N_ACTIONS = len(ACTIONS)
    DIRECTION_TO_VECTOR = {
        0: (-1, 0),
        1: (0, -1),
        2: (1, 0),
        3: (0, 1)
    }

    OBJECTS = {184: 2, 400: 2, 616: 2, 832: 2, 941: 3, 443: 5, 418: 5, 455: 1, 1038: 3, 392: 6, 608: 6, 824: 6, 1040: 6, 501: 6, 717: 6, 933: 6}
    SWORDS = {2: 1084, 5: 514, 1: 559, 6: 319, 3: 1221}

    def __init__(
            self,
            seed: int = None,
            max_states: int = 10000,
            max_state_actions: int = 100000,
            n_monitor: int = 100,
            **kwargs
    ):
        self._seed = seed
        self._gen = np.random.default_rng(seed) if 'gen' not in kwargs else kwargs['gen']
        self._Q = ActionValue(self.N_ACTIONS, seed=seed, scale=2.0, offset=-1.0) if 'Q' not in kwargs else kwargs['Q']
        self._state_counter = VisitsCounter(max_states) if 'state_counter' not in kwargs else kwargs['state_counter']
        self._state_action_counter = VisitsCounter(max_state_actions) if 'state_action_counter' not in kwargs else kwargs['state_action_counter']
        self._result_tracker = ResultTracker(n_monitor) if 'result_tracker' not in kwargs else kwargs['result_tracker']

    @property
    def n_state_(self):
        return len(self._state_counter)

    @property
    def n_state_action_(self):
        return len(self._state_action_counter)

    @property
    def result_(self):
        return self._result_tracker

    def _mask_action(self, link: np.ndarray, objs: np.ndarray):
        mask = np.zeros(self.N_ACTIONS, dtype=float)

        cx, cy, sword, direction = link
        dx, dy = self.DIRECTION_TO_VECTOR[direction]
        fx, fy = cx + dx, cy + dy
        cur_obj = objs[cx, cy]
        front_obj = objs[fx, fy]

        # Unable to stay
        mask[0] = -1e5

        # Unable to move forward if there are wall and monsters in front
        if front_obj in (0, 7, 11, 12, 13):
            mask[3] = -1e5

        # Unable to pick up if there is no sword here
        if cur_obj != 6:
            mask[4] = -1e5

        # Unable to drop sword if the link has no sword or there is a sword here
        if sword == 0 or cur_obj == 6:
            mask[5] = -1e5

        # Unable to attack if link has no sword or there is no monster in front
        if sword == 0 or front_obj not in (7, 11, 12, 13):
            mask[6] = -1e5

        return mask

    def preprocess(self, obs):
        link = obs['link']
        tiles = obs['tiles']

        sword = {
            k: -1 for k in self.SWORDS.keys()
        }
        objects = {
            k: -1 for k in self.OBJECTS.keys()
        }
        maps = np.zeros(shape=(36, 36)) - 1

        for x, y, obj_id, prop in tiles:
            if obj_id in (0, 6, 7, 11, 12, 13):
                maps[x, y] = obj_id

            if obj_id == 6:
                sword[prop] = y * 36 + x
            elif obj_id in (7, 11, 12, 13):
                objects[y * 36 + x] = prop

        mask = self._mask_action(link, maps)
        state = (int(link[1] * 36 + link[0]), int(link[2]), int(link[3]) , *sword.values(), *objects.values())
        return state, mask

    def end_episode(self, reward, death, timeout, cleared):
        self._result_tracker.update(reward, death, timeout, cleared)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        rand = {
            'seed': self._seed,
            'state': self._gen.bit_generator.state
        }

        with open(os.path.join(path, 'rand.json'), mode='w') as f:
            json.dump(rand, f)

        with open(os.path.join(path, 'Q.pkl'), mode='wb') as f:
            pickle.dump(self._Q, f)

        with open(os.path.join(path, 'state.pkl'), mode='wb') as f:
            pickle.dump(self._state_counter, f)

        with open(os.path.join(path, 'state-action.pkl'), mode='wb') as f:
            pickle.dump(self._state_action_counter, f)

        with open(os.path.join(path, 'result.pkl'), mode='wb') as f:
            pickle.dump(self._result_tracker, f)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, 'rand.json'), mode='r') as f:
            rand = json.load(f)

        with open(os.path.join(path, 'state.pkl'), mode='rb') as f:
            state_counter = pickle.load(f)

        with open(os.path.join(path, 'state-action.pkl'), mode='rb') as f:
            state_action_counter = pickle.load(f)

        with open(os.path.join(path, 'Q.pkl'), mode='rb') as f:
            Q = pickle.load(f)

        with open(os.path.join(path, 'result.pkl'), mode='rb') as f:
            result_tracker = pickle.load(f)

        gen = np.random.default_rng(rand['seed'])
        gen.bit_generator.state = rand['state']

        return ZeldaQAgent(
            seed=rand['seed'],
            gen=gen,
            state_counter=state_counter,
            state_action_counter=state_action_counter,
            result_tracker=result_tracker,
            Q=Q
        )

    def act(self, observation: any, info: dict):
        state, mask = self.preprocess(observation)
        Q = self._Q[state] + mask
        return np.argmax(Q)

    def update(self, state, action, reward, next_state, next_mask, done, gamma, alpha):
        target = reward if done else reward + gamma * self._Q[next_state][np.argmax(self._Q[next_state] + next_mask)]
        self._Q[state][action] = self._Q[state][action] + alpha * (target - self._Q[state][action])

        self._state_counter[state] += 1
        self._state_action_counter[(action, *state)] += 1

    def eps_greedy(self, epsilon, state, mask):
        if self._gen.random() < epsilon:
            p = np.where(mask == -1e5, 0.0, 1.0 / (self.N_ACTIONS - np.count_nonzero(mask)))
            action = self._gen.choice(self.ACTIONS, p=p)
        else:
            Q = self._Q[state] + mask
            action = np.argmax(Q)
        return action

    def reward_func(self, state, action, next_state, death, timeout, is_cleared):
        if is_cleared:
            return 1.0
        elif death:
            return -1.0
        elif timeout:
            return -1.0

        reward = -.001
        if (action, *state) not in self._state_action_counter:
            reward += 0.25
        else:
            reward += 0.25 / self._state_action_counter[(action, *state)]

        if next_state not in self._state_counter:
            reward += 0.25
        else:
            reward += 0.25 / np.sqrt(self._state_counter[next_state])
        return reward


def best_play(path: str):
    agent = ZeldaQAgent.load(path)
    kym.evaluate(
        env_id='kymnasium/ZeldaAdventure-Stage-3',
        agent=agent,
        render_mode='human',
        bgm=True
    )


def learn(
        path_agent,
        max_episode=100000,
        full_exp_episode=1000,
        init_epsilon=1.0,
        min_epsilon=0.1,
        decay_rate=0.9995,
        gamma=0.995,
        alpha=0.1,
        max_steps=1000,
        save_interval=500,
        seed=None
):
    env = gym.make(
        'kymnasium/ZeldaAdventure-Stage-3',
        render_mode='none',
    )
    agent = None
    if os.path.exists(path_agent):
        try:
            agent = ZeldaQAgent.load(path_agent)
        except:
            pass

    if agent is None:
        agent = ZeldaQAgent(
            seed,
            max_states=100000,
            max_state_actions=100000 * ZeldaQAgent.N_ACTIONS
        )

    pbar = tqdm(range(max_episode), desc='Episode')
    epsilon = init_epsilon

    for i in pbar:
        epsilon = max(epsilon * decay_rate, min_epsilon) if i > full_exp_episode else epsilon
        total_reward, steps = 0.0, 0

        death, timeout, cleared = False, False, False
        done = death or timeout or cleared

        obs, _ = env.reset()
        state, mask = agent.preprocess(obs)

        while not done:
            if steps <= 300:
                epsilon = 0.05
            else:
                epsilon = 0.2
            action = agent.eps_greedy(epsilon, state, mask)
            next_obs, _, terminated, truncated, _ = env.step(action)
            next_state, next_mask = agent.preprocess(next_obs)
            steps += 1

            death = terminated or truncated
            timeout = steps >= max_steps
            cleared = next_obs['link'][0] == 22 and next_obs['link'][1] == 1
            reward = agent.reward_func(state, action, next_state, death, timeout, cleared)
            agent.update(state, action, reward, next_state, next_mask, done, gamma, alpha)
            obs, state, mask = next_obs, next_state, next_mask

            total_reward += reward

            done = death or timeout or cleared

        avg_reward = total_reward / steps

        agent.end_episode(avg_reward, death, timeout, cleared)

        if agent.result_.n_episode_ % save_interval == 0:
            agent.save(os.path.join(path_agent, f'./Ep. #{agent.result_.n_episode_}'))

        pbar.set_postfix(
            eps=f'{epsilon:.5f}',
            n_state=agent.n_state_,
            n_state_action=agent.n_state_action_,
            episode=agent.result_.n_episode_,
            steps=steps,
            reward=f'{agent.result_.mean_rewards_:.5f}',
            best=f'{agent.result_.best_rewards_:.5f}',
            death_ratio=f'{agent.result_.death_ratio_:.5f}',
            timeout_ratio=f'{agent.result_.timeout_ratio_:.5f}',
            cleared_ratio=f'{agent.result_.cleared_ratio_:.5f}',
        )

if __name__ == '__main__':
    learn(
        './zelda',
        max_episode=5000000,
        full_exp_episode=0,
        init_epsilon=0.2,
        min_epsilon=0.20,
        decay_rate=0.99995,
        gamma=0.99995,
        max_steps=999,
        save_interval=2000,
        alpha=0.1
    )
