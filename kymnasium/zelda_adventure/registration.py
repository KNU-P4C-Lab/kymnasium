import os
import gymnasium as gym
from .env import ZeldaAdventureEnv
from ..util import play_bgm, ObsType


_BGM_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'bgm.ogg')


def _create_env(
        max_steps: int,
        stage: int,
        bgm: bool = False,
        obs_type: ObsType = 'custom',
        **kwargs
) -> gym.Env:
    if bgm:
        play_bgm(_BGM_PATH)

    env = ZeldaAdventureEnv(
        max_steps=max_steps,
        stage=stage,
        **kwargs
    )

    return env