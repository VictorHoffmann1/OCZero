import gymnasium as gym
import numpy as np


class BaseWrapper(gym.Wrapper):
    def __init__(self, env, clip_reward):
        super().__init__(env)
        self.clip_reward = clip_reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        info["raw_reward"] = reward
        if self.clip_reward:
            reward = np.sign(reward)

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def close(self):
        return self.env.close()
