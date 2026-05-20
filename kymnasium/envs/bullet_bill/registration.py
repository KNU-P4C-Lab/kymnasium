import gymnasium as gym
from .env import BulletBillEnv
from .consts import ASSET_DIR
from ...common.util import play_bgm
from ...common.types import ObsType


_BGM_PATH = ASSET_DIR / 'bgm.ogg'


def _create_env(
        game_duration: float = 180,
        init_spawn_interval: float = 1,
        min_spawn_interval: float = 0.2,
        max_spawns: int = 30,
        max_spawn_duration: float = 150,
        bgm: bool = False,
        obs_type: ObsType = 'default',
        continuous_action: bool = False,
        stage: int = 1,
        seed: int | None = None,
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
        obs_type=obs_type,
        seed=seed,
        **kwargs
    )

    return env