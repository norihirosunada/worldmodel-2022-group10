import random
import math
import numpy as np
import copy
import ast

import torch
from torch import nn
from torch.nn import functional as F

class Agent:
    def __init__(self):

    def __call__(self, obs, training=True):

class ValueModel(nn.Module):
    """
    Universal Value Function Approximator: UVFA
    V(s, t) -> s: state representation, t: task representation
    """
    def __init__(self, state_dim, hidden_dim, act=F.elu, task_dim=4):
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


def ActionModel(nn.Module):
    """
    Determine action from state and task representation
    """
    def __init__(self, state_dim, )


