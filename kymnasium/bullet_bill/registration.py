import os
import gymnasium as gym
from .env import BulletBillEnv
from .wrappers import RGBImgObsWrapper
from ..util import play_bgm, ObsType


_BGM_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'bgm.ogg')


def _create_env(
        game_duration: float = 180,
        init_spawn_interval: float = 1,
        min_spawn_interval: float = 0.2,
        max_spawns: int = 30,
        max_spawn_duration: float = 150,
        bgm: bool = False,
        obs_type: ObsType = 'custom',
        continuous_action: bool = False,
        stage: int = 1,
        **kwargs
) -> gym.Env:
    if bgm:
        play_bgm(_BGM_PATH)

    env = BulletBillEnv(
        game_duration=game_duration,
        init_spawn_interval=init_spawn_interval,
        min_spawn_interval=min_spawn_interval,
        max_spawns=max_spawns,
        max_spawn_duration=max_spawn_duration,
        continuous_action=continuous_action,
        stage=stage,
        **kwargs
    )

    if obs_type == 'image':
        env = RGBImgObsWrapper(env)
    
    return env