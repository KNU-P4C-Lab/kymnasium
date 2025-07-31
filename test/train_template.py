import gymnasium as gym
import kymnasium as kym


class YourAgent(kym.Agent):
    def act(self, observation: any, info: dict):
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        pass

    def save(self, path: str):
        pass


def train():
    env = gym.make(
        id='kymnasium/environment-id-here',
        render_mode='human', # "human" for display the game; "rgb_array" for background rendering,
        obs_type='custom', # "custom" for tabular representation of observation; "image" for image representation
        bgm=True # True for playing background music; False, otherwise
    )
    agent = YourAgent()
    done = False
    observation, info = env.reset()

    while not done:
        action = agent.act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        '''
        Write some codes to train your agent here.
        '''
