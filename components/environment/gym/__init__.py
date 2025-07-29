from ..base import BaseWrapper


class GymWrapper(BaseWrapper):
    """
    Make your own wrapper: Atari Wrapper
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, _, done, info = self.env.step(action)
        info["raw_reward"] = reward
        return obs, reward, done, info

    def reset(
        self,
    ):
        obs, info = self.env.reset()

        return obs
