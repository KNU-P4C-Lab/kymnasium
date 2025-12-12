from .env import AlkkagiEnv
from ..evaluate import RemoteEnvWrapper
from typing import Any


class AlkkagiRemoteEnvWrapper(RemoteEnvWrapper):
    def serialize(self, observation: Any):
        return observation

    def verify_action(self, action) -> bool:
        env = self.env.unwrapped
        return isinstance(env, AlkkagiEnv) and action['turn'] == env.turn_

