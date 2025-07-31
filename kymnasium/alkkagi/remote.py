from .env import AlkkagiEnv
from kymnasium.evaluate import RemoteEnvWrapper


class AlkkagiRemoteEnvWrapper(RemoteEnvWrapper):
    def serialize(self, observation: any):
        return observation

    def verify_action(self, action) -> bool:
        env = self.env.unwrapped
        return isinstance(env, AlkkagiEnv) and action['turn'] == env.turn_

