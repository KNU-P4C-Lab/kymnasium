from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, observation: any, info: dict):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'Agent':
        raise NotImplementedError()