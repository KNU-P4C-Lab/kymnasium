from .env import AlkkagiEnv
from .wrappers import RGBImgObsWrapper
from .consts import ASSET_DIR
from ...common.util import play_bgm
from ...common.types import ObsType


_BGM_PATH = ASSET_DIR / 'bgm.ogg'

def _create_env(
        n_stones: int,
        n_obstacles: int,
        bgm: bool = False,
        obs_type: ObsType = 'default',
        **kwargs
):
    if bgm:
        play_bgm(_BGM_PATH)

    env = AlkkagiEnv(
        n_stones=n_stones,
        n_obstacles=n_obstacles,
        obs_type=obs_type,
        **kwargs
    )

    if obs_type == 'image':
        env = RGBImgObsWrapper(env)

    return env
