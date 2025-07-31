import os
import gymnasium as gym
from .env import AvoidBlurpEnv
from .wrappers import RGBImgObsWrapper
from kymnasium.util import play_bgm, ObsType


_BGM_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'bgm.ogg')


def _create_env(
        game_duration: float = 180,
        init_spawn_interval: float = 1,
        min_spawn_interval: float = 0.2,
        max_spawns: int = 30,
        prob_spawn_on_player: float = 0.0,
        max_spawn_duration: float = 150,
        bgm: bool = False,
        obs_type: ObsType = 'custom',
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
        **kwargs
    )

    if obs_type == 'image':
        env = RGBImgObsWrapper(env)
    
    return env