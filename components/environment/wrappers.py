from typing import Any, SupportsFloat, Tuple, Dict
from components.encoders.object_centric_encoder import ObjectCentricEncoder
from components.environment.game import Game
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None  # type: ignore[assignment]

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
        venv,
        max_objects,
        speed_scale=10.0,
        use_rgb=False,
        use_category=False,
    ):
        super().__init__(venv)
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

    def reset(self):
        _ = self.venv.reset()
        return self.encoder(self.venv)

    def step_wait(self):
        images, rewards, terminated, infos = self.venv.step_wait()
        encoded_obs = self.encoder(self.venv)

        # Process terminal observations in infos to match the encoded observation space
        if isinstance(infos, list):
            for i, info in enumerate(infos):
                if "terminal_observation" in info:
                    # The terminal observation should also be encoded to match our observation space
                    # Since we can't encode a single terminal observation without the environment state,
                    # we'll use the current encoded observation as a reasonable approximation
                    info["terminal_observation"] = encoded_obs[i]
                info["image"] = images[i]  # type: ignore[assignment]
        elif isinstance(infos, dict):
            if "terminal_observation" in infos:
                # The terminal observation should also be encoded to match our observation space
                # Since we can't encode a single terminal observation without the environment state,
                # we'll use the current encoded observation as a reasonable approximation
                infos["terminal_observation"] = encoded_obs
            infos["image"] = images

        return encoded_obs, rewards, terminated, infos


class StickyActionEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sticky action.

    Paper: https://arxiv.org/abs/1709.06009
    Official implementation: https://github.com/mgbellemare/Arcade-Learning-Environment

    :param env: Environment to wrap
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self._sticky_action = 0  # NOOP
        return self.env.reset(**kwargs)

    def step(self, action: int) -> AtariStepReturn:
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        return self.env.step(self._sticky_action)


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


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.
    Also handles firing when a life is lost (when used with EpisodicLifeEnv).

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]
        self._last_lives = 0
        self._fire_action = 1

    def step(self, action: int) -> AtariStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if we need to fire after a life loss
        current_lives = info.get("lives", 0)
        if (
            current_lives > 0  # Still have lives left
            and current_lives < self._last_lives
        ):  # Life was lost
            # Fire to start the new life
            obs, _, terminated, truncated, info = self.env.step(self._fire_action)
            if terminated or truncated:
                # If firing caused game over, step with no-op to get stable state
                obs, _, terminated, truncated, info = self.env.step(0)

        self._last_lives = current_lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(self._fire_action)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)

        # Initialize life tracking
        _, _, _, _, info = self.env.step(0)  # Get current state info
        self._last_lives = info.get("lives", 0)

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


class OCAtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Termination signal when a life is lost.
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    :param action_repeat_probability: Probability of repeating the last action
    :param warp_frame: If True (default), the frame is resized to 84x84 and converted to grayscale.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        max_pool: bool = True,
        frame_skip: int = 1,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
        time_limit: int = 2000,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip, max_pool=max_pool)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if time_limit > 0:
            env = TimeLimit(env, time_limit)

        super().__init__(env)


class AtariWrapper(Game):
    def __init__(self, env, obs_shape, discount: float):
        """Atari Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        """
        super().__init__(env, obs_shape, env.action_space.n, discount)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def get_max_episode_steps(self):
        return self.env.get_max_episode_steps()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation

    def close(self):
        self.env.close()
