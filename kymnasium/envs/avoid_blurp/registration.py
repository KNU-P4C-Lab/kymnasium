import os
from typing import Literal
import gymnasium as gym
from .env import AvoidBlurpEnv
from ...common.util import play_bgm
from ...common.types import ObsType


_BGM_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'bgm.ogg')


def _create_env(
        game_duration: float = 180,
        init_spawn_interval: float = 1,
        min_spawn_interval: float = 0.2,
        max_spawns: int = 30,
        prob_spawn_on_player: float = 0.0,
        max_spawn_duration: float = 150,
        bgm: bool = False,
        obs_type: ObsType = 'default',
        continuous_action: bool = False,
        mode: Literal['vertical', 'ballistic'] = 'vertical',
        stage: int = 1,
        **kwargs
) -> gym.Env:
    if bgm:
        play_bgm(_BGM_PATH)

    env = AvoidBlurpEnv(
        game_duration=game_duration,
        init_spawn_interval=init_spawn_interval,
        min_spawn_interval=min_spawn_interval,
        max_spawns=max_spawns,
        prob_spawn_on_player=prob_spawn_on_player,
        max_spawn_duration=max_spawn_duration,
        continuous_action=continuous_action,
        mode=mode,
        stage=stage,
        obs_type=obs_type,
        **kwargs
    )
    return env