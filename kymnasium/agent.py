from abc import ABC, abstractmethod
from typing import Any, Dict


class Agent(ABC):
    @abstractmethod
    def act(self, observation: Any, info: Dict):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'Agent':
        raise NotImplementedError()