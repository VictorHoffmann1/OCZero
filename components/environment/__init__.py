import gymnasium as gym
from .gym import GymWrapper
from .atari import AtariWrapper
from .dmc import DMCWrapper
from .wrapper import (
    ObjectCentricEncoderWrapper,
    TimeLimit,
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    Monitor,
)
import random
from ocatari.core import OCAtari

try:
    from dm_env import specs
except ImportError:
    specs = None
try:
    import dmc2gym
except ImportError:
    dmc2gym = None


def make_envs(game_setting, game_name, num_envs, seed, save_path=None, **kwargs):
    assert game_setting in ["Atari", "DMC", "Gym"]
    if game_setting == "Atari":
        _env_fn = make_atari
    elif game_setting == "Gym":
        _env_fn = make_gym
    elif game_setting == "DMC":
        _env_fn = make_dmc
    else:
        raise NotImplementedError()

    if game_setting == "DMC":
        seed = random.randint(1, 1000)

    envs = [
        _env_fn(
            game_name,
            seed=i + seed,
            # seed=seed,
            save_path=save_path,
            **kwargs,
        )
        for i in range(num_envs)
    ]
    return envs


def make_env(game_setting, game_name, num_envs, seed, save_path=None, **kwargs):
    assert game_setting in ["Atari", "DMC", "Gym"]
    if game_setting == "Atari":
        _env_fn = make_atari
    elif game_setting == "Gym":
        _env_fn = make_gym
    elif game_setting == "DMC":
        _env_fn = make_dmc
    else:
        raise NotImplementedError()

    seed = random.randint(1, 1000)

    env = _env_fn(game_name, seed=seed, save_path=save_path, **kwargs)

    return env


def make_atari(game_name, seed, save_path=None, **kwargs):
    """Make Atari games
    Parameters
    ----------
    game_name: str
        name of game (Such as Breakout, Pong)
    kwargs: dict
        skip: int
            frame skip
        obs_shape: (int, int)
            observation shape
        gray_scale: bool
            use gray observation or rgb observation
        seed: int
            seed of env
        max_episode_steps: int
            max moves for an episode
        save_path: str
            the path of saved videos; do not save video if None
            :param seed:
    """
    # params
    env_id = game_name + "NoFrameskip-v4"  # TODO: Implement for v5 versions maybe?
    skip = kwargs["n_skip"] if kwargs.get("n_skip") else 4
    max_episode_steps = (
        kwargs["max_episode_steps"]
        if kwargs.get("max_episode_steps")
        else 108000 // skip
    )
    max_objects = kwargs.get("max_objects", 32)
    max_pool = False
    episodic_life = kwargs.get("episodic_life")
    clip_reward = kwargs.get("clip_reward")

    env_kwargs = {
        "mode": "ram",
        "hud": False,
        "obs_mode": "ori",
    }

    # env = gym.make(env_id)
    env = OCAtari(env_id, **env_kwargs)
    # set seed
    env.reset(seed=seed)

    # random restart
    env = NoopResetEnv(env, noop_max=30)
    # frame skip
    env = MaxAndSkipEnv(env, skip=skip, max_pool=max_pool)
    # episodic trajectory
    if episodic_life:
        env = EpisodicLifeEnv(env)
    # set max limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    # Object Centric Encoder
    env = ObjectCentricEncoderWrapper(env, max_objects=max_objects, speed_scale=8.0)
    # save video to given
    if save_path:
        env = Monitor(env, filename=str(save_path), override_existing=True)

    # your wrapper
    env = AtariWrapper(env, clip_reward=clip_reward)
    return env


def make_gym(game_name, seed, save_path=None, **kwargs):
    save_path = kwargs.get("save_path")
    obs_to_string = kwargs.get("obs_to_string")
    skip = kwargs["n_skip"] if kwargs.get("n_skip") else 4

    env = gym.make(game_name)
    env = GymWrapper(env, obs_to_string=obs_to_string)

    # frame skip
    env = MaxAndSkipEnv(env, skip=skip)

    # set seed
    env.seed(seed)

    # save video to given
    if save_path:
        env = Monitor(env, directory=str(save_path), force=True)

    env = GymWrapper(env, obs_to_string=obs_to_string)
    return env


def make_dmc(game_name, seed, save_path=None, **kwargs):
    """Make Atari games
    Parameters
    ----------
    game_name: str
        name of game (Such as Breakout, Pong)
    kwargs: dict
        image_based: bool
            observation is image or state

    """
    # params
    if "CMU" in game_name:
        domain_name, task_name = game_name.rsplit("_", 1)
    else:
        domain_name, task_name = game_name.split("_", 1)
    image_based = kwargs.get("image_based")
    obs_shape = kwargs["obs_shape"] if kwargs.get("obs_shape") else [3, 96, 96]
    skip = kwargs["n_skip"] if kwargs.get("n_skip") else 2
    max_episode_steps = kwargs["max_episode_steps"] // skip
    clip_reward = kwargs.get("clip_reward")
    obs_to_string = kwargs.get("obs_to_string")
    # fix the bug of env (from the paper DrQv2)
    camera_id = 2 if "quadruped" in domain_name else 0

    # # make env
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=image_based,
        height=obs_shape[1] if image_based else 96,
        width=obs_shape[1] if image_based else 96,
        frame_skip=skip,
        channels_first=False,
        camera_id=camera_id,
        # time_limit=max_episode_steps,
    )

    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # save video to given
    if save_path:
        env = Monitor(env, directory=str(save_path), force=True)

    # your wrapper
    env = DMCWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)
    return env
