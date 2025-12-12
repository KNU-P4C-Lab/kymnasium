import os
from .env import AlkkagiEnv
from .wrappers import RGBImgObsWrapper
from ..util import play_bgm, ObsType


_BGM_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'bgm.ogg')

def _create_env(
        n_stones: int,
        n_obstacles: int,
        bgm: bool = False,
        obs_type: ObsType = 'custom',
        **kwargs
):
    if bgm:
        play_bgm(_BGM_PATH)

    env = AlkkagiEnv(
        n_stones=n_stones,
        n_obstacles=n_obstacles,
        **kwargs
    )

    if obs_type == 'image':
        env = RGBImgObsWrapper(env)

    return env