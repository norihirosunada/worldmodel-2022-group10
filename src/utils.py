import time
import tqdm
import os
import gc
from collections import deque

import gym
import pybullet_envs

import pathlib

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


"""
今回のReplayBufferの定義
"""
class ReplayBuffer(object):
    def __init__(self, capacity, observation_shape, action_dim):
        # length of chunk
        self.capacity = capacity

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool_)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        """Add an experiences to replay buffer
        Args:
            observation:
            action:
            reward:
            done:

        Returns:
        """

        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        if self.index == self.capacity -1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        """Sample an experience from replay buffer.
        Args:
            batch_size:
            chunk_length:

        Returns:
            experiences -> Array: [batch_size, chunk_length, dimensions] for each of {observation, action, reward, dones}
        """
        assert len(self) > 0, "Replay buffer is empty."
        assert len(self) > chunk_length

        # index of episode finished
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):

            #  Checking chunks to see if they cross episodes
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders, episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:]
        )
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1]
        )
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1
        )

        return sampled_observations, sampled_actions, sampled_rewards, sampled_done


    def __len__(self):
        return self.capacity if self.is_filled else self.index


def preprocess_obs(obs):
    """Transform observations
    Args:
        obs: observations
    Returns:
        transformed observations
    """
    obs = obs.astype(np.float32)
    normalized_obs = obs / 255. - 0.5
    return normalized_obs


def lambda_target(rewards, values, gamma, lambda_):
    """Calculate λ-return to update value function
    Args:
        rewards:
        values:
        gamma:
        lambda_:

    Returns:
    """
    V_lambda = torch.zeros_like(rewards, device=rewards.device)

    H = rewards.shape[0] - 1
    V_n = torch.zeros_like(rewards, device=rewards.device)
    V_n[H] = values[H]
    for n in range(1, H+1):
        V_n[:-n] = (gamma ** n) * values[n:]
        for k in range(1, n+1):
            if k == n:
                V_n[:-n] += V_n[:-n] + (gamma ** (n-1)) * rewards[k:]
            else:
                V_n[:-n] += (gamma ** (k-1)) * rewards[k:-n+k]

        if n == H:
            V_lambda += (lambda_ ** (H-1)) * V_n
        else:
            V_lambda += (1 - lambda_) * (lambda_ ** (n - 1)) * V_n

    return V_lambda