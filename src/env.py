import numpy as np
import gym
from gym import spaces


def make_env():
    env = GridWorld()
    return env


# custom 2d grid world environment which extends gym.Env
class GridWorld(gym.Env):
    """
        - a size x size grid world which agent can ba at any cell other than terminal cell
        - terminal cell is set to be the last cell or bottom right cell in the grid world
        - 5x5 grid world example where X is the agent location and O is the tremial cell
          .....
          .....
          ..X..
          .....
          ....O -> this is the terminal cell where this is agent headed to
        - Reference : https://github.com/openai/gym/blob/master/gym/core.py
    """
    metadata = {'render.modes': ['console']}

    # actions available
    UP   = 0
    LEFT = 1
    DOWN = 2
    RIGHT= 3

    def __init__(self, size=10):
        """
        Args:
            size: size of the grid world
        """
        super(GridWorld, self).__init__()

        self.start_state = None
        self.agent_position = None
        self.end_state = None

        assert size%2 == 0, "Maze size must be an even number!"
        self.size      = size + 1  # actual environment has separate wall.

        # respective actions of agents : up, down, left and right
        self.action_space = spaces.Discrete(4)
        self.holes = [[2, 5], [5, 2], [8, 5], [5, 8]]

        # set the observation space to (1,) to represent agent position in the grid world
        # staring from [0,size*size)
        self.observation_space = spaces.Box(low=0, high=size*size, shape=(1,), dtype=np.uint8)
        self.is_task_setup = False

    def set_task(self, start_pos, end_pos):
        """
        Args:
            start_pos: A tuple which specify start position of the agent
            end_pos: A tuple which specify end position of the agent
        """
        start_pos = self.convert_pos_to_idx(start_pos)
        end_pos = self.convert_pos_to_idx(end_pos)
        self.start_state = start_pos
        self.agent_position = start_pos
        self.end_state = end_pos
        self.is_task_setup = True

    def step(self, action):
        info = {} # additional information

        goal_reward = 10
        punish_reward = -0.01

        row  = self.agent_position // self.size
        col  = self.agent_position % self.size
        assert self.size >= row >= 0 or self.size >= col >= 0, "agent out of maze"
        room = self.get_room_id(row, col)
        # print(room)

        # assert row != self.size//2, "agent is in wall"
        # assert col != self.size//2, "agent is in wall"

        if action == self.UP:
            if row != 0 and row - 1 != self.size//2:
                self.agent_position -= self.size
            elif [row - 1, col] in self.holes:
                self.agent_position -= self.size * 2
                # print("get into")
        elif action == self.LEFT:
            if col != 0 and col - 1 != self.size//2:
                self.agent_position -= 1
            elif [row, col-1] in self.holes:
                self.agent_position -= 2
                # print("get into")
        elif action == self.DOWN:
            if row != self.size - 1 and row + 1 != self.size//2:
                self.agent_position += self.size
            elif [row + 1, col] in self.holes:
                self.agent_position += self.size * 2
                # print("get into")
        elif action == self.RIGHT:
            if col != self.size - 1 and col + 1 != self.size//2:
                self.agent_position += 1
            elif [row, col + 1] in self.holes:
                self.agent_position += 2
                # print("get into")
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        done   = bool(self.agent_position == self.end_state)

        # reward agent when it is in the terminal cell, else reward = 0
        reward = goal_reward if done else punish_reward

        return np.array([self.agent_position]).astype(np.uint8), reward, done, info

    def render(self, mode='console'):
        '''
            render the state
        '''
        if mode != 'console':
          raise NotImplementedError()

        row  = self.agent_position // self.size
        col  = self.agent_position % self.size

        for r in range(self.size):
            for c in range(self.size):
                if r == row and c == col:
                    print("X",end='')
                elif r == self.size//2 or c == self.size//2:
                    print("■", end='')
                else:
                    print('.',end='')
            print('')
        print('------------')

    def reset(self):
        # -1 to ensure agent inital position will not be at the end state
        self.agent_position = self.start_state

        return np.array([self.agent_position]).astype(np.uint8)

    def close(self):
        pass

    def get_room_id(self, row, col):
        """
        Return the room id where current agent stay

        Upper left: 0
        Upper right: 1
        Lower left: 2
        Lower right: 3
        """
        if row in range(0, self.size//2):
            if col in range(0, self.size//2):
                return 0
            else:
                return 1
        else:
            if col in range(0, self.size//2):
                return 2
            else:
                return 3

    def convert_idx_to_pos(self, pos_index: int) -> tuple:
        """位置のindexをrowとcolのタプルに変換
        Args:
            pos_index: 位置のindex
        Returns:
            positions: tuple of position, (row, col)
        """
        row = pos_index//self.size
        col = pos_index%self.size
        return row, col

    def convert_pos_to_idx(self, pos) -> int:
        """Convert position(tuple-format) to index format
        Args:
            pos: tuple -> (row, col)
        Returns:
            pos(index): int
        """
        row, col = pos[0], pos[1]
        return int(row * self.size + col)

    def convert_task_to_pos(self, task):
        """Convert task representations with [start_index, end_index] format to array format
        Args:
            task: [start_index, end_index]
        Returns:
            [(row_start, col_start), (row_end, col_end)]
        """
        return [self.convert_idx_to_pos(task[0]), self.convert_idx_to_pos(task[1])]

    @staticmethod
    def manhattan_distance(start, goal):
        """開始地点から終了地点までのマンハッタン距離を測定
        Args:
            start: tuple of start position (row, col)
            goal: tuple of end position
        Returns:
            manhattan_dist
        """
        manhattan_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        return manhattan_dist