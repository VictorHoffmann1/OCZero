# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from components.agents.models.relational_network import RelationalNetwork


def mlp(
    input_size,
    hidden_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ELU,
    init_zero=False,
):
    """
    MLP layers
    :param input_size:
    :param hidden_sizes:
    :param output_size:
    :param output_activation:
    :param activation:
    :param init_zero:   bool, zero initialization for the last layer (including w and b).
                        This can provide stable zero outputs in the beginning.
    :return:
    """
    sizes = [input_size] + hidden_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [
                nn.Linear(sizes[i], sizes[i + 1]),
                nn.BatchNorm1d(sizes[i + 1]),
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


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        action_space_size,
        hidden_dim,
        feature_dim=6,
        is_continuous=False,
    ):
        """
        Dynamics network
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_space_size = action_space_size

        self.action_embedding = nn.Linear(action_space_size, input_dim)

        self.next_object_state_predictor = nn.Sequential(
            nn.Linear(feature_dim + action_space_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, state, action):
        # encode action
        if self.is_continuous:
            raise NotImplementedError("Continuous action space not supported yet.")
        else:
            action_one_hot = torch.nn.functional.one_hot(
                action.flatten().long(), num_classes=self.action_space_size
            )
            state = state + self.action_embedding(action_one_hot.float())
            next_state = self.next_hidden_state_predictor(state)

        return next_state


class ValuePolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        fc_layers,
        value_output_size,
        policy_output_size,
        init_zero,
        is_continuous=False,
        policy_distribution="beta",
        **kwargs,
    ):
        super().__init__()

        if is_continuous:
            raise NotImplementedError("Continuous action space not supported yet.")

        self.v_num = kwargs.get("v_num")

        self.fc_backbone = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim),
        )

        self.fc_values = nn.ModuleList(
            [
                mlp(
                    input_dim,
                    fc_layers,
                    value_output_size,
                    init_zero=False if is_continuous else init_zero,
                )
                for _ in range(self.v_num)
            ]
        )
        self.fc_policy = mlp(
            input_dim,
            fc_layers if not is_continuous else [64],
            policy_output_size,
            init_zero=init_zero,
        )

        self.input_dim = input_dim
        self.is_continuous = is_continuous
        self.init_std = 1.0
        self.min_std = 0.1

    def forward(self, x):
        x = self.fc_backbone(x)

        values = []
        for i in range(self.v_num):
            value = self.fc_values[i](x)
            values.append(value)

        policy = self.fc_policy(x)

        if self.is_continuous:
            action_space_size = policy.shape[-1] // 2
            policy[:, :action_space_size] = 5 * torch.tanh(
                policy[:, :action_space_size] / 5
            )  # soft clamp mu
            policy[:, action_space_size:] = (
                torch.nn.functional.softplus(
                    policy[:, action_space_size:] + self.init_std
                )
                + self.min_std
            )  # .clip(0, 5)  # same as Dreamer-v3

        return torch.stack(values), policy


class SupportNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        fc_layers,
        output_support_size,
        init_zero,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.fc = mlp(input_dim, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x):
        x = self.fc(x)
        return x


class SupportLSTMNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        fc_layers,
        output_support_size,
        lstm_hidden_size,
        init_zero,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_size)
        self.bn_reward_sum = nn.BatchNorm1d(lstm_hidden_size)
        self.fc = mlp(
            lstm_hidden_size, fc_layers, output_support_size, init_zero=init_zero
        )

    def forward(self, x, hidden):
        x, hidden = self.lstm(x.unsqueeze(0), hidden)
        x = x.squeeze(0)
        x = self.bn_reward_sum(x)
        x = nn.functional.relu(x)
        x = self.fc(x)
        return x, hidden


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.input_dim = input_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.layer(x)


class ProjectionHeadNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.layer(x)
