import random
import math
import numpy as np
import copy
import ast

import torch
from torch import nn
from torch.nn import functional as F


class Teacher(object):
    def __init__(self, state_shape, task_dim, state_dim=256, hidden_dim=256):
        super(Teacher, self).__init__()
        self.encoder = Encoder(state_shape, state_dim)
        self.value_model = ValueModel(state_dim, hidden_dim)
        self.action_model = ActionModel(state_dim, hidden_dim, task_dim)

        self.device = next(self.action_model.parameters()).device

    def __call__(self, state, task, training=True):
        """
        Args:
            state: student's state-value -> (1, 1, H, W)
            task: 4-dim task representation [r_start, c_start, r_end, c_end]
            training:
        Returns:
            action: agent's action on current state
        """
        state_embedd = self.encoder(state)
        state_embedd = torch.as_tensor(state_embedd, device=self.device)
        task = torch.as_tensor(task, device=self.device)

        with torch.no_grad():
            action_probs = self.action_model(state_embedd, task, training=training)

        action = torch.argmax(action_probs)

        return action.cpu().numpy()


class Encoder(nn.Module):
    """
    State encoder for student value table.
    """
    def __init__(self, state_shape, state_dim, act=F.relu):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[1], 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, state_dim, 3, 2)

        self.act = act

    def forward(self, state):
        """
        Args:
            state: 2d representation of student state (B, C, H, W)
        Returns:
            embedd_state: (B, state_dim)
        """
        hidden = self.act(self.conv1(state))
        hidden = self.act(self.conv2(hidden))
        hidden = self.act(self.conv3(hidden))
        hidden = self.act(self.conv4(hidden))
        hidden = self.act(self.conv5(hidden))

        return torch.nn.AvgPool2d(hidden.shape[-2:])(hidden)


class ValueModel(nn.Module):
    """
    Universal Value Function Approximator: UVFA
    V(s, t) -> s: state representation, t: task representation
    """
    def __init__(self, state_dim, hidden_dim, task_dim=4, act=F.elu):
        super(ValueModel, self).__init__()
        self.fc1_s = nn.Linear(state_dim, hidden_dim)
        self.fc2_s = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_t = nn.Linear(task_dim, hidden_dim)
        self.fc2_t = nn.Linear(hidden_dim, hidden_dim)
        self.act = act

        # Linear transform after concatenate task and state
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, task):
        hidden_s = self.act(self.fc1_s(state))
        hidden_s = self.act(self.fc2_s(hidden_s))

        hidden_t = self.act(self.fc1_t(task))
        hidden_t = self.act(self.fc2_t(hidden_t))

        # concatenate task and state embeddings
        hidden = torch.concatenate((hidden_s, hidden_t), dim=1)
        hidden = self.act(self.fc3(hidden))
        state_value = self.act(self.fc4(hidden))

        return state_value


class ActionModel(nn.Module):
    """
    Determine action from state and task representation
    """
    def __init__(self, state_dim, hidden_dim, action_dim, task_dim=4, act=F.elu):
        super(ActionModel, self).__init__()
        self.fc1_s = nn.Linear(state_dim, hidden_dim)
        self.fc2_s = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_t = nn.Linear(task_dim, hidden_dim)
        self.fc2_t = nn.Linear(hidden_dim, hidden_dim)
        self.act = act

        # Linear transform after concatenate task and state
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, task):
        """Predict action which maximize estimated value with ValueModel
        Args:
            state: student's q-table embeddings -> (B, state_dim)
            task: task representation -> (B, 4)
        Returns:
            Probability for each action
        """
        hidden_s = self.act(self.fc1_s(state))
        hidden_s = self.act(self.fc2_s(hidden_s))

        hidden_t = self.act(self.fc1_t(task))
        hidden_t = self.act(self.fc2_t(hidden_t))

        # concatenate task and state embeddings
        hidden = torch.concatenate((hidden_s, hidden_t), dim=1)
        hidden = self.act(self.fc3(hidden))
        action_prob = nn.Softmax()(self.fc4(hidden))

        return action_prob




