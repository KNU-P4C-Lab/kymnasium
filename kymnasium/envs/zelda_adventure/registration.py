import gymnasium as gym
from .consts import ASSET_DIR
from .env import ZeldaAdventureEnv
from ...common.util import play_bgm
from ...common.types import ObsType


_BGM_PATH = ASSET_DIR /  'bgm.ogg'


def _create_env(
        max_steps: int,
        stage: int,
        bgm: bool = False,
        obs_type: ObsType = 'default',
        **kwargs
) -> gym.Env:
    if bgm:
        play_bgm(_BGM_PATH)

    env = ZeldaAdventureEnv(
        max_steps=max_steps,
        stage=stage,
        obs_type=obs_type,
        **kwargs
    )

    return env
