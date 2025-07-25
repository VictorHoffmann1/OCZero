import math
import torch

import numpy as np
import torch.nn as nn

from components.model.base_model import BaseNet, renormalize
from components.encoders.relational_network import RelationalNetwork


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    momentum=0.1,
    init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [
                nn.Linear(sizes[i], sizes[i + 1]),
                nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                act(),
            ]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        top_k=64,
    ):
        """Representation network"""

        super().__init__()
        self.relational_net = RelationalNetwork(input_dim, hidden_dim, top_k)

    def forward(self, x):
        """Forward pass of the representation network"""
        x = self.relational_net(x)
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


# Predict next hidden states gesiven current state and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        reward_dim,
        fc_reward_layers,
        full_support_size,
        action_space_size,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        input_dim: int
            dim of flatten hidden states
        reward_dim: int
            dim of reward prediction
        action_space_size: int
            number of actions in the environment
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.action_embedding = nn.Linear(action_space_size, input_dim)

        self.next_hidden_state_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim),
        )

        self.reward_projector = nn.Sequential(
            nn.Linear(input_dim, reward_dim),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(input_size=reward_dim, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(
            self.lstm_hidden_size,
            fc_reward_layers,
            full_support_size,
            init_zero=init_zero,
            momentum=momentum,
        )

    def forward(self, x, action_one_hot, reward_hidden):
        # Add action embedding to the hidden states
        x = x + self.action_embedding(action_one_hot)  # (B, input_dim)
        # Predict next hidden state
        x = self.next_hidden_state_predictor(x)  # (B, input_dim)
        next_state = x
        # Reward and value prefix prediction
        x = self.reward_projector(x)  # (B, reward_dim)
        value_prefix, reward_hidden = self.lstm(x.unsqueeze(0), reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return next_state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        dynamic_mean = []

        for name, param in self.action_embedding.named_parameters():
            dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        for name, param in self.next_hidden_state_predictor.named_parameters():
            dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = (
            self.reward_projector[0].weight.detach().cpu().numpy().reshape(-1)
        )

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        action_space_size,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        momentum=0.1,
        init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        input_dim: int
            dim of hidden state
        action_space_size: int
            action space
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        """
        super().__init__()

        self.fc_value = mlp(
            input_dim,
            fc_value_layers,
            full_support_size,
            init_zero=init_zero,
            momentum=momentum,
        )
        self.fc_policy = mlp(
            input_dim,
            fc_policy_layers,
            action_space_size,
            init_zero=init_zero,
            momentum=momentum,
        )

    def forward(self, x):
        # Let's keep it simple
        # input x: (B, H)
        value = self.fc_value(x)
        policy = self.fc_policy(x)

        # NOTE: If the performance is not good, we can try further processing before the fc layers
        return policy, value


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        reward_dim,
        top_k,
        action_space_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        reward_support_size,
        value_support_size,
        inverse_value_transform,
        inverse_reward_transform,
        lstm_hidden_size,
        bn_mt=0.1,
        proj_hid=64,  # Try higher values if it doesn't work well
        proj_out=64,
        pred_hid=32,
        pred_out=64,
        init_zero=False,
        state_norm=False,
    ):
        """EfficientZero network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        action_space_size: int
            action space
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        value_support_size: int
            dim of value output
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        bn_mt: float
            Momentum of BN
        proj_hid: int
            dim of projection hidden layer
        proj_out: int
            dim of projection output layer
        pred_hid: int
            dim of projection head (prediction) hidden layer
        pred_out: int
            dim of projection head (prediction) output layer
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        state_norm: bool
            True -> normalization for hidden states
        """
        super(EfficientZeroNet, self).__init__(
            inverse_value_transform, inverse_reward_transform, lstm_hidden_size
        )
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.action_space_size = action_space_size

        self.representation_network = RepresentationNetwork(
            input_dim,
            hidden_dim,
            top_k,
        )

        self.dynamics_network = DynamicsNetwork(
            hidden_dim,
            reward_dim,
            fc_reward_layers,
            reward_support_size,
            action_space_size,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.prediction_network = PredictionNetwork(
            hidden_dim,
            action_space_size,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = torch.nn.functional.one_hot(
            action.long().flatten(), num_classes=self.action_space_size
        ).float()  # (B, action_space_size)
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(
            encoded_state, action_one_hot, reward_hidden
        )

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    def get_params_mean(self):
        representation_mean = self.representation_network.get_param_mean()
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()
