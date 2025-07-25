import torch

from components.config import BaseConfig
from components.environment.transforms import Transforms
from components.model.model import EfficientZeroNet
from components.environment.wrappers import (
    OCAtariWrapper,
    AtariWrapper,
    ObjectCentricEncoderWrapper,
)
from ocatari.core import OCAtari
import time


class AtariConfig(BaseConfig):
    def __init__(self):
        super(AtariConfig, self).__init__(
            training_steps=100000,
            last_steps=20000,
            test_interval=10000,
            log_interval=1000,
            vis_interval=1000,
            test_episodes=32,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=10000,
            max_moves=12000,
            test_max_moves=12000,
            history_length=400,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=50,
            batch_size=256,
            td_steps=5,
            num_actors=1,
            # network initialization/ & normalization
            episode_life=True,
            init_zero=True,
            clip_reward=True,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=100000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=2,
            total_transitions=100 * 1000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=4,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            # network parameters
            hidden_dim=64,
            reward_dim=64,
            top_k=16,
            # reward sum
            lstm_hidden_size=128,
            lstm_horizon_len=5,
            # siamese
            proj_hid=256,
            proj_out=256,
            pred_hid=128,
            pred_out=256,
        )
        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        # TODO: Add these params in config
        self.max_objects = 32

        self.start_transitions = self.start_transitions * 1000 // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.resnet_fc_reward_layers = [
            32
        ]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [
            32
        ]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [
            32
        ]  # Define the hidden layers in the policy head of the prediction network

    def visit_softmax_temperature_fn(self, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        game = self.new_game(
            save_video=save_video,
            save_path=save_path,
            video_callable=video_callable,
        )
        self.action_space_size = game.action_space_size
        self.obs_shape = game.obs_shape

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape[-1],
            self.hidden_dim,
            self.reward_dim,
            self.top_k,
            self.action_space_size,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm,
        )

    def new_game(
        self,
        seed=None,
        save_video=False,
        save_path=None,
        video_callable=None,
        uid=None,
        test=False,
    ):
        wrapper_kwargs = {
            "noop_max": 0 if test or "v5" in self.env_name else 30,
            "terminal_on_life_loss": self.episode_life,
            "frame_skip": self.frame_skip if "v4" in self.env_name else 1,
            "max_pool": False,
            "time_limit": self.max_moves,
        }
        env_kwargs = {
            "mode": "ram",
            "hud": False,
            "obs_mode": "ori",
        }
        if "v5" in self.env_name:
            env_kwargs["frameskip"] = self.frame_skip
            env_kwargs["repeat_action_probability"] = 0.25
            env_kwargs["full_action_space"] = True

        env = OCAtari(self.env_name, **env_kwargs)
        env = OCAtariWrapper(env, **wrapper_kwargs)
        env = ObjectCentricEncoderWrapper(
            env, max_objects=self.max_objects, speed_scale=8.0
        )  # TODO: Add these params in config
        self.obs_shape = env.obs_shape
        if seed is not None:
            env.reset(seed=seed)

        if save_video:
            print(f"Saving video to {save_path} with uid {uid}...")
            from gymnasium.wrappers import Monitor

            env = Monitor(
                env,
                directory=save_path,
                force=True,
                video_callable=video_callable,
                uid=uid,
            )
        return AtariWrapper(env, obs_shape=self.obs_shape, discount=self.discount)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            raise NotImplementedError("Not implemented yet")
            self.transforms = Transforms(
                self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2])
            )

    def transform(self, images):
        raise NotImplementedError("Not implemented yet")
        return self.transforms.transform(images)


game_config = AtariConfig()
