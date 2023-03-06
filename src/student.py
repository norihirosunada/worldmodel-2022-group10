import random
import math
import numpy as np
import copy
import ast


class QLearningAgent:
    """
        Q学習 エージェント
    """

    def __init__(
            self,
            env,
            alpha=.2,
            epsilon=.1,
            gamma=.99,
            actions=None,
            observation=None,
    ):

        self.alpha = alpha
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.q_values = self._init_q_values()
        self.q_table = np.zeros((self.env.size ** 2, len(self.actions)))
        self.state_values = None

        assert env.is_task_setup, "Please specify task by calling env.set_task()!"

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def transfer_q_values(self, state):
        """Initialize Q-table by state.
        Args:
            state: (env.size * env.size, )
        Returns:
        """
        for i in range(0, len(state)):
            key = str([i])
            self.q_values[key] = state[i]

    def init_state(self):
        """
            状態の初期化
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):
        # ε-greedy選択
        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:   # greedy 行動
            action_list = [i for i, x in enumerate(self.q_values[self.state]) if x == max(self.q_values[self.state])] # 最大値が複数ある時のため
            action = random.choice(action_list)

        self.previous_action = action
        return action

    def observe(self, next_state, reward=None, training=True):
        """
            次の状態と報酬の観測
        """
        next_state = str(next_state)
        if next_state not in self.q_values:  # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if reward is not None:
            self.reward_history.append(reward)
            if training:
                self.learn(reward)

    def learn(self, reward):
        """
            Q値の更新
        """
        q = self.q_values[self.previous_state][self.previous_action]  # Q(s, a)
        max_q = max(self.q_values[self.state])  # max Q(s')
        self.q_values[self.previous_state][self.previous_action] = q + \
            self.alpha * (reward + (self.gamma * max_q) - q)

    def dict_to_table(self, q_dict: dict) -> np.ndarray:
        """Convert the format of q values to 2d-array of state-values.
        Args:
            q_dict: q-values(dict format)

        Returns:
            q_values:
            state_values: state_value-table(table format)
        """

        for key, value in q_dict.items():
            idx = ast.literal_eval(key)[0]
            self.q_table[idx] = value
        self.state_values = np.reshape(np.sum(self.q_table, axis=1), (int(math.sqrt(self.q_table.shape[0])), -1))

