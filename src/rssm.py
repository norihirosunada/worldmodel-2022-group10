import torch
from torch.distributions import Normal
from torch import nn
from torch.nn import functional as F



class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, rnn_hidden_dim, hidden_dim=200, min_stddev=0.1, act=F.relu):
        super(TransitionModel, self).__init__()

        # 状態，行動，RNNの隠れ層の次元を決める
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)

        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_rnn_hidden_embedded_obs = nn.Linear(rnn_hidden_dim + 1024, hidden_dim)

        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)

        # next hidden stateを計算
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act

    def forward(self, state, action, rnn_hidden, embedded_next_obs):
        """
        h_t+1 = f(h_t, s_t, a_t)
        prior p(s_t+1 | h_t+1) と posterior q(s_t+1 | h_t+1, e_t+1) を返す
        この2つが近づくように学習する
        """
        next_state_prior, rnn_hidden = self.prior(self.recurrent(state, action, rnn_hidden))
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior, rnn_hidden

    def recurrent(self, state, action, rnn_hidden):
        """
        h_t+1 = f(h_t, s_t, a_t)を計算する
        """
        hidden = self.act(self.fc_state_action(torch.cat([state, action], dim=1)))
        #h_t+1を求める
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        return rnn_hidden

    def prior(self, rnn_hidden):
        """
        prior p(s_t+1 | h_t+1) を計算する
        """
        #h_t+1を求める
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(self.fc_state_stddev_prior(hidden)) + self._min_stddev
        return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        posterior q(s_t+1 | h_t+1, e_t+1)  を計算する
        """
        # h_t+1, o_t+1を結合し, q(s_t+1 | h_t+1, e_t+1) を計算する
        hidden = self.act(self.fc_rnn_hidden_embedded_obs(torch.concat([rnn_hidden, embedded_obs], dim=1)))
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)


class ObservationModel(nn.Module):
    """
    p(o_t | s_t, h_t)
    低次元の状態表現から画像を再構成するデコーダ (3, 64, 64)
    """
    def __init__(self, state_dim, rnn_hidden_dim):
        super(ObservationModel, self).__init__()
        self.fc = nn.Linear(state_dim + rnn_hidden_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)


    def forward(self, state, rnn_hidden):
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs


class RewardModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    低次元の状態表現から報酬を予測する
    """
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=F.elu):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        reward = self.fc4(hidden)
        return reward


class RSSM:
    def __init__(self, state_dim, action_dim, rnn_hidden_dim, device):
        self.transition = TransitionModel(state_dim, action_dim, rnn_hidden_dim).to(device)
        self.observation = ObservationModel(state_dim, rnn_hidden_dim, ).to(device)
        self.reward = RewardModel(state_dim, rnn_hidden_dim, ).to(device)