import warnings

import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari


class ObjectCentricEncoder:
    """Rule-based encoder for Atari games with batched environment support."""

    IMG_WIDTH = 160  # Default Atari image width
    IMG_HEIGHT = 210  # Default Atari image height

    def __init__(
        self,
        max_objects: int = 64,
        speed_scale: float = 10.0,
        use_rgb: bool = False,
        use_category: bool = False,
    ):
        self.speed_scale = speed_scale
        self.n_features = 6
        if use_rgb:
            self.n_features += 3
        if use_category:
            self.n_features += 3
        self.max_objects = max_objects
        self.use_rgb = use_rgb
        self.use_category = use_category

        # Add caching for object extraction to avoid repeated wrapper traversal
        self._env_ocatari_cache = {}

    def __call__(self, env) -> np.ndarray:
        """Encodes a batch of frames into feature spaces by return paddle position and ball position + velocity.
        Args:
            states: Batch of input frames [num_envs, H, W, C]
        Returns:
            np.ndarray: Batch of encoded features [num_envs, feature_dim]
        """
        objects = self.get_ocatari_objects(env)

        features = np.zeros(
            (self.max_objects, self.n_features)
        )  # Initialize features for each env
        idx = 0
        for object in objects:
            if object.category != "NoObject":
                if idx >= self.max_objects:
                    warnings.warn(
                        f"More objects than max_objects ({self.max_objects}) in environment. Truncating.",
                        UserWarning,
                    )
                    break
                object_vector = np.array(
                    [
                        self.normalize(object.center[0], self.IMG_WIDTH, True),
                        self.normalize(object.center[1], self.IMG_HEIGHT, True),
                        self.normalize(object.dx, self.speed_scale, False),
                        self.normalize(object.dy, self.speed_scale, False),
                        self.normalize(object.w, self.IMG_WIDTH, False),
                        self.normalize(object.h, self.IMG_HEIGHT, False),
                    ]
                )
                if self.use_rgb:
                    rgb = object.rgb
                    rgb_vector = np.array(
                        [
                            self.normalize(rgb[0], 255, False),
                            self.normalize(rgb[1], 255, False),
                            self.normalize(rgb[2], 255, False),
                        ]
                    )
                    object_vector = np.concatenate((object_vector, rgb_vector))
                if self.use_category:
                    category_vector = np.zeros(3)
                    if object.category == "Player":
                        category_vector[0] = 1.0
                    elif object.category == "Ball":
                        category_vector[1] = 1.0
                    else:
                        category_vector[2] = 1.0
                    object_vector = np.concatenate((object_vector, category_vector))
                features[idx] = object_vector
                idx += 1

        return features

    def get_ocatari_objects(self, env: gym.Env):
        """
        Get the objects from the underlying OCAtari environment through the wrapper chain.
        Uses caching to avoid repeated wrapper traversal.

        :param env: The wrapped environment
        :return: The objects from the OCAtari environment
        """
        env_id = id(env)

        # Check cache first
        if env_id in self._env_ocatari_cache:
            return self._env_ocatari_cache[env_id].objects

        # Traverse through wrappers to find the OCAtari instance
        current_env = env
        while hasattr(current_env, "env"):
            if isinstance(current_env, OCAtari):
                self._env_ocatari_cache[env_id] = current_env
                return current_env.objects
            current_env = current_env.env

        # Check if the current env is OCAtari
        if isinstance(current_env, OCAtari):
            self._env_ocatari_cache[env_id] = current_env
            return current_env.objects

        # If we can't find OCAtari, raise an error
        raise ValueError("No OCAtari environment found in the wrapper chain")

    @staticmethod
    def normalize(value, scale, centering=False):
        """
        Normalize a value to the range [-1, 1] based on a scale.

        :param value: The value to normalize
        :param scale: The scale for normalization
        :param centering: If True, center the value around 0 before scaling
        :return: Normalized value in the range [-1, 1]
        """
        if centering:
            return np.clip(2 * (value / scale) - 1, -1, 1) if scale != 0 else 0
        else:
            return np.clip(value / scale, -1, 1) if scale != 0 else 0
