import os
import numpy as np
import gymnasium as gym
from .env import GridSurvivorEnv
from .wrappers import FullyObsWrapper, RGBImgObsWrapper
from ..util import play_bgm, ObsType


_BGM_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'bgm.ogg')

def _create_env(
        max_steps: int,
        blueprint: str,
        max_hit_points: int,
        damage: int,
        bgm: bool = False,
        obs_type: ObsType = 'custom',
        **kwargs
) -> gym.Env:
    if bgm:
        play_bgm(_BGM_PATH)

    blueprint = np.loadtxt(blueprint, dtype=str, delimiter=',').T
    env = GridSurvivorEnv(
        max_steps=max_steps,
        blueprint=blueprint,
        max_hit_points=max_hit_points,
        damage=damage,
        **kwargs
    )

    if obs_type == 'image':
        env = RGBImgObsWrapper(env)
    else:
        env = FullyObsWrapper(env)

    return env

