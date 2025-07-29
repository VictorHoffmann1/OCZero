from typing import Any, SupportsFloat, Tuple, Dict, Optional, Union
import csv
import json
import os
import cv2
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
import numpy as np
import time
from components.environment.object_centric_encoder import ObjectCentricEncoder

AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]


class ObjectCentricEncoderWrapper(gym.Wrapper):
    """
    Wrapper to encode the observations using a rule-based encoder.
    :param venv: Environment to wrap
    :param encoder: Encoder class/function to apply to each observation
    :param n_features: Number of features in the encoded observation
    """

    def __init__(
        self,
        env,
        max_objects,
        speed_scale=10.0,
        use_rgb=False,
        use_category=False,
    ):
        super().__init__(env)
        self.encoder = ObjectCentricEncoder(
            max_objects=max_objects,
            speed_scale=speed_scale,
            use_rgb=use_rgb,
            use_category=use_category,
        )
        self.obs_shape = (
            max_objects,
            self.encoder.n_features,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )

    def reset(self, seed=None, **kwargs):
        _ = self.env.reset(seed=seed, **kwargs)
        return self.encoder(self.env)

    def step(self, action: int):
        image, rewards, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        encoded_obs = self.encoder(self.env)

        if "terminal_observation" in info:
            # The terminal observation should also be encoded to match our observation space
            # Since we can't encode a single terminal observation without the environment state,
            # we'll use the current encoded observation as a reasonable approximation
            info["terminal_observation"] = encoded_obs
            info["image"] = image

        return encoded_obs, rewards, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, terminated, truncated, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            terminated = True
            info["TimeLimit.truncated"] = True
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info: dict = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> AtariStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info["lives"]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = info["lives"]
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    and optionally return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    :param max_pool: If True, return the max over the last two frames. If False, return the last frame only.
    """

    def __init__(self, env: gym.Env, skip: int = 4, max_pool: bool = True) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, (
            "No dtype specified for the observation space"
        )
        assert env.observation_space.shape is not None, (
            "No shape defined for the observation space"
        )
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
        self._skip = skip
        self._max_pool = max_pool

    def step(self, action: int) -> AtariStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations (if enabled).

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        if self._max_pool:
            max_frame = self._obs_buffer.max(axis=0)
            return max_frame, total_reward, terminated, truncated, info
        else:
            return self._obs_buffer[1], total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human", **kwargs):
        raise NotImplementedError()
        img = self.max_frame
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class DMC_Obs_Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        obs = np.moveaxis(obs, 0, -1)
        return obs


class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: tuple[str, ...] = (),
        info_keywords: tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env=env)
        self.t_start = time.time()
        self.results_writer = None
        if filename is not None:
            env_id = env.spec.id if env.spec is not None else None
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=reset_keywords + info_keywords,
                override_existing=override_existing,
            )

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards: list[float] = []
        self.needs_reset = True
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []
        self.total_steps = 0
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info: dict[str, Any] = {}

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(
                    f"Expected you to pass keyword argument {key} into reset"
                )
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(float(reward))
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super().close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> list[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> list[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> list[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times


class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class

    :param filename: the location to save a log file. When it does not end in
        the string ``"monitor.csv"``, this suffix will be appended to it
    :param header: the header dictionary object of the saved csv
    :param extra_keys: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    def __init__(
        self,
        filename: str = "",
        header: Optional[dict[str, Union[float, str]]] = None,
        extra_keys: tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        if header is None:
            header = {}
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        filename = os.path.realpath(filename)
        # Create (if any) missing filename directories
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Append mode when not overriding existing file
        mode = "w" if override_existing else "a"
        # Prevent newline issue on Windows, see GH issue #692
        self.file_handler = open(filename, f"{mode}t", newline="\n")
        self.logger = csv.DictWriter(
            self.file_handler, fieldnames=("r", "l", "t", *extra_keys)
        )
        if override_existing:
            self.file_handler.write(f"#{json.dumps(header)}\n")
            self.logger.writeheader()

        self.file_handler.flush()

    def write_row(self, epinfo: dict[str, float]) -> None:
        """
        Write row of monitor data to csv log file.

        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()
