# KNU Gymnasium, or Kymnasium for Reinforcement Learning Environments

Welcome to Kymnasium! This project is a collection of reinforcement learning environments developed for the Artificial Intelligence course at the Department of Computer Science and Engineering, Kangwon National University.

## About the Project
Kymnasium provides a simple and effective platform for students to learn, implement, and test reinforcement learning algorithms. The environments are designed to be straightforward, allowing you to focus on the core concepts of RL. This project is inspired by [Gymnasium](https://gymnasium.farama.org/).

## Environments
* **Alkkagi**: A Korean traditional game where the objective is to flick your stones to knock the opponent's stones off the board.
* **Avoid Blurp**: An environment where the Mario must move left and right to avoid free-falling Blurps.
* **Bullet Bill**: An environment where the Mario must jump around to avoid flying Bullet Bills.
* **Grid World**: A classic grid-world environment where the agent navigates a maze to reach a goal.
* **Zelda's Adventure**: A game where the Link must navigate a maze and fight against enemies to reach a goal.


## Getting Started
### Installation
```bash
pip -U install kymnasium
```

### Implement Your Agent
To train your own agent, you need to override 'kymnasium.Agent' and implement three methods as below:

```python

import kymnasium as kym


# Your agent logic goes here
class YourAgent(kym.Agent):
    def act(self, observation: any, info: dict):
        # Replace this with your agent's action selection logic return env.action_space.sample()
        pass

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        # Load a pre-trained agent
        pass

    def save(self, path: str):
        # Save the trained agent
        pass
```
### Basic Training loop
```python
import gymnasium as gym


# Train the agent for 100 episodes
EPISODES = 100

# Path for saving your agent
PATH_AGENT = 'agent.pkl'
agent = YourAgent()

# Create the environment
env = gym.make(
    id="kymnasium/GridWorld-Crossing-26x26", # Environment ID
    render_mode='rgb_array', # or 'human',
    obs_type='custom', # or 'image'
    bgm=False # or True for playing background music
)
for _ in range(EPISODES):
    observation, info = env.reset()
    done = False
    
    while not done: 
        action = agent.act(observation, info) 
        observation, reward, terminated, truncated, info = env.step(action) 
        done = terminated or truncated
        # Here writes any training logic
        
    # Close the environment        
    env.close()

# Save your agent
agent.save(PATH_AGENT)
```

### Live Evaluation of Your Agent

```python

import kymnasium as kym


agent = YourAgent.load(PATH_AGENT)

kym.evaluate(
    env_id='kymnasium/GridWorld-Crossing-26x26',
    agent=agent,
    render_mode='human',
    bgm=True
)
```

## Manual Play
If you want to manually play the environment, see below:
```python
from kymnasium.grid_world import ManualPlayWrapper


agent = ManualPlayWrapper(
    env_id='kymnasium/GridWorld-Crossing-26x26',
    bgm=True,
    debug=True
)
agent.play()
```
