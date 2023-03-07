import time
import datetime
import tqdm
import os
import gc

import pathlib
import argparse
import yaml
import numpy as np

import torch
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from env_gym import make_env
from utils import ReplayBuffer, preprocess_obs, clip_grad_norm_, lambda_target
from rssm import RSSM
from dreamer import Encoder, ValueModel, ActionModel, Agent


def make_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str, help='Path to config file')
    return parser


def create_log_dir(parent_dir):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = parent_dir / f'{now}'
    log_dir.mkdir(exist_ok=True, parents=True)
    
    return log_dir


if __name__ == '__main__':

    parser = make_argparse()
    args = parser.parse_args()

    with open(args.config, mode='r') as config_file:
        config = yaml.safe_load(config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make environment
    name = 'HalfCheetahBulletEnv-v0'
    env = make_env(name)

    # Get environment settings
    observation_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config['buffer']['capacity'],
        observation_shape=observation_shape,
        action_dim=action_dim
    )
    
    # Create summary writer for log to TensorBoard
    parent_dir = pathlib.Path(os.getcwd()).parent / pathlib.Path('log')
    log_dir = create_log_dir(parent_dir)
    writer = SummaryWriter(log_dir)

    # Create models
    state_dim = config['model']['state_dim']
    rnn_hidden_dim = config['model']['rnn_hidden_dim']
    
    encoder = Encoder().to(device)
    rssm = RSSM(
        state_dim=state_dim,
        action_dim=action_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        device=device
    )
    value_model = ValueModel(
        state_dim=state_dim,
        rnn_hidden_dim=rnn_hidden_dim
    ).to(device)
    action_model = ActionModel(
        state_dim=state_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        action_dim=action_dim
    ).to(device)

    # Aggregate dynamics model parameters
    model_params = (
        list(encoder.parameters()) +
        list(rssm.transition.parameters()) +
        list(rssm.observation.parameters()) +
        list(rssm.reward.parameters())
    )

    # Set optimizers
    dynamics_lr = float(config['model']['learning_rate']['dynamics'])
    value_lr = float(config['model']['learning_rate']['value'])
    action_lr = float(config['model']['learning_rate']['action'])
    epsilon = float(config['model']['epsilon'])

    model_optimizer = torch.optim.Adam(model_params, lr=dynamics_lr, eps=epsilon)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=value_lr, eps=epsilon)
    action_optimizer = torch.optim.Adam(action_model.parameters(), lr=action_lr, eps=epsilon)

    # Collect experiences from random action
    env = make_env(name)
    for episode in tqdm.tqdm(range(config['model']['seed_episodes'])):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs
    del env
    gc.collect()

    # Start entire learning
    for episode in range(config['model']['seed_episodes'], config['model']['all_episodes']):
        # 1. Collect experiences
        start = time.time()
        policy = Agent(encoder, rssm.transition, action_model)
        env = make_env(name)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(obs)
            # Add gaussian noise for exploration
            action += np.random.normal(0, np.sqrt(config['agent']['action_noise_var']), env.action_space.shape[0])
            next_obs, reward, done, _ = env.step(action)

            # Save {obs, action, reward, done} to replay buffer
            replay_buffer.push(obs, action, reward, done)

            obs = next_obs
            total_reward += reward

        # Write reward and elapsed time to Tensorboard
        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, config['model']['all_episodes'], total_reward))
        print('elapsed time for interaction: %.2fs' % (time.time() - start))

        # 2. Update NN parameters
        start = time.time()
        chunk_length = config['training']['chunk_length']
        batch_size = config['training']['batch_size']
        
        for update_step in range(config['model']['collect_interval']):
            # 2.1 Dynamics learning
            observations, actions, rewards, _ = \
                replay_buffer.sample(batch_size, chunk_length)

            observations = preprocess_obs(observations)
            observations = torch.as_tensor(observations, device=device)

            # (B, L, H, W, C) -> (B, L, C, H, W). L: Chunk length
            observations = observations.transpose(3, 4).transpose(2, 3)
            # (B, L, H, W, C) -> (L, B, C, H, W)
            observations = observations.transpose(0, 1)

            # (B, L, Action_dim) -> (L, B, Action_dim)
            # (B, L, 1) -> (L, B, 1)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)

            # (L, B, C, H, W) -> (L*B, C, H, W) -> (L, B, Encoded_dim)
            embedded_obs = encoder(
                observations.reshape(-1, 3, 64, 64)
            ).view(chunk_length, batch_size, -1)

            states = torch.zeros(chunk_length, batch_size, state_dim, device=device)
            rnn_hiddens = torch.zeros(chunk_length, batch_size, rnn_hidden_dim, device=device)

            # Initialize init state(s_0) with 0
            state = torch.zeros(states.shape[1:], device=device)
            rnn_hidden = torch.zeros(rnn_hiddens.shape[1:], device=device)

            # Compute KL divergence between prior and posterior
            kl_loss = 0
            for l in range(chunk_length - 1):
                next_state_prior, next_state_posterior, rnn_hidden = \
                    rssm.transition(state, actions[l], rnn_hidden, embedded_obs[l + 1])
                state = next_state_posterior.rsample()
                states[l + 1] = state
                rnn_hiddens[l + 1] = rnn_hidden

                # kl -> (batch_size, state_dim)
                kl = kl_divergence(next_state_posterior, next_state_prior)
                kl_loss += kl.clamp(min=config['training']['free_nats']).mean()
            kl_loss /= (chunk_length - 1)

            # Discord stats[0] and rnn_hidden[0] because these are initialized 0.
            states = states[1:]
            rnn_hiddens = rnn_hiddens[1:]

            # Reconstruct observations and Predict rewards
            # L = chunk length - 1

            # -> (L*B, state_dim)
            flatten_states = states.view(-1, state_dim)
            # -> (L*B, rnn_hidden_dim)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, rnn_hidden_dim)
            # -> (L, B, C, H, W)
            recon_observations = \
                rssm.observation(flatten_states, flatten_rnn_hiddens).view(chunk_length - 1, batch_size, 3, 64, 64)
            # -> -> (L, B, 1)
            predicted_rewards = \
                rssm.reward(flatten_states, flatten_rnn_hiddens).view(chunk_length - 1, batch_size, 1)

            # Calculate prediction error with observations and rewards
            obs_loss = 0.5 * F.mse_loss(recon_observations, observations[1:], reduction='none').mean([0, 1]).sum()
            reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

            # Update nn parameters(dynamics)
            model_loss = kl_loss + obs_loss + reward_loss
            model_optimizer.zero_grad()
            model_loss.backward(retain_graph=True)
            clip_grad_norm_(model_params, config['training']['clip_grad_norm'])
            model_optimizer.step()


            # 2.2 Behavior learning(Actor-critic)
            # detach parameters of transition_model
            imagination_horizon = config['training']['imagination_horizon']
            flatten_states = flatten_states.detach()
            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

            # Prepare a Tensor to hold future state predictions several steps ahead
            imaginated_states = torch.zeros(imagination_horizon + 1, *flatten_states.shape, device=flatten_states.device)
            imaginated_rnn_hiddens = torch.zeros(imagination_horizon + 1, *flatten_rnn_hiddens.shape, device=flatten_rnn_hiddens.device)

            # Before creating imaginated trajectory, the initial state is the one we just used to update model parameters.
            # State on observations sampled from replay buffer
            imaginated_states[0] = flatten_states
            imaginated_rnn_hiddens[0] = flatten_rnn_hiddens

            # Create imaginated trajectory
            for h in range(1, imagination_horizon + 1):
                actions = action_model(flatten_states, flatten_rnn_hiddens)
                flatten_states_prior, flatten_rnn_hiddens = \
                    rssm.transition.prior(rssm.transition.recurrent(flatten_states, actions, flatten_rnn_hiddens))
                flatten_states = flatten_states_prior.rsample()
                imaginated_states[h] = flatten_states
                imaginated_rnn_hiddens[h] = flatten_rnn_hiddens

            # Calculate rewards and values on imaginated trajectory.
            flatten_imaginated_states = imaginated_states.view(-1, state_dim)
            flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(-1, rnn_hidden_dim)
            imaginated_rewards = \
                rssm.reward(flatten_imaginated_states,
                            flatten_imaginated_rnn_hiddens).view(imagination_horizon + 1, -1)
            imaginated_values = \
                value_model(flatten_imaginated_states,
                            flatten_imaginated_rnn_hiddens).view(imagination_horizon + 1, -1)

            lambda_target_values = lambda_target(imaginated_rewards, imaginated_values, config['agent']['gamma'], config['agent']['lambda_'])

            # Update action model(Actor)
            action_loss = -1 * lambda_target_values.mean()
            action_optimizer.zero_grad()
            action_loss.backward()
            clip_grad_norm_(action_model.parameters(), config['training']['clip_grad_norm'])
            action_optimizer.step()

            # Update value model(Critic)
            imaginated_values = value_model(flatten_imaginated_states.detach(), flatten_imaginated_rnn_hiddens.detach()).view(imagination_horizon + 1, -1)
            value_loss = 0.5 * F.mse_loss(imaginated_values, lambda_target_values.detach())
            value_optimizer.zero_grad()
            value_loss.backward()
            clip_grad_norm_(value_model.parameters(), config['training']['clip_grad_norm'])
            value_optimizer.step()

            # Output log to TensorBoard
            print('update_step: %3d model loss: %.5f, kl_loss: %.5f, '
                  'obs_loss: %.5f, reward_loss: %.5f, '
                  'value_loss: %.5f action_loss: %.5f'
                  % (update_step + 1, model_loss.item(), kl_loss.item(),
                     obs_loss.item(), reward_loss.item(),
                     value_loss.item(), action_loss.item()))
            total_update_step = episode * config['model']['collect_interval'] + update_step
            writer.add_scalar('model loss', model_loss.item(), total_update_step)
            writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('reward loss', reward_loss.item(), total_update_step)
            writer.add_scalar('value loss', value_loss.item(), total_update_step)
            writer.add_scalar('action loss', action_loss.item(), total_update_step)

        print('elapsed time for update: %.2f[s]' % (time.time() - start))
        # 3. Evaluate once every {test_interval} episodes.
        if (episode + 1) % config['model']['test_interval'] == 0:
            policy = Agent(encoder, rssm.transition, action_model)
            start = time.time()
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = policy(obs, training=False)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            writer.add_scalar('total reward at test', total_reward, episode)
            print('Total test reward at episode[%4d/%4d]: %f' % (episode + 1, config['model']['all_episodes'], total_reward))
            print('Elapsed time: %.2f[s]' % (time.time() - start))

        # 4. Save once every {model_save_interval} episodes.
        if (episode + 1) % config['model']['model_save_interval'] == 0:
            model_log_dir = log_dir / f'episode_{(episode + 1):04d}'
            model_log_dir.mkdir(exist_ok=True)
            torch.save(encoder.state_dict(), model_log_dir / 'encoder.pth')
            torch.save(rssm.transition.state_dict(), model_log_dir / 'rssm.pth')
            torch.save(rssm.observation.state_dict(), model_log_dir / 'obs_model.pth')
            torch.save(value_model.state_dict(), model_log_dir / 'value_model.pth')
            torch.save(action_model.state_dict(), model_log_dir / 'action_model.pth')
        del env
        gc.collect()
    writer.close()


