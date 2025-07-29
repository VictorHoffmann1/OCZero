from ..base import BaseWrapper


class AtariWrapper(BaseWrapper):
    """
    Make your own wrapper: Atari Wrapper
    """

    def __init__(self, env, clip_reward=False):
        super().__init__(env, clip_reward)
