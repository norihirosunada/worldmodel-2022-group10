import numpy as np
import gym
from gym import spaces


def make_env(start_pos, end_pos):
    env = GridWorld(start_pos, end_pos)
    env.reset()
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

    def __init__(self, start_pos, end_pos, size=10):
        """
        Args:
            start_pos: start position of the agent
            end_pos: end position of the agent
            size: size of the grid world
        """
        super(GridWorld, self).__init__()

        assert size%2 == 0, "Maze size must be an even number!"
        self.size      = size + 1  # actual environment has separate wall.
        self.end_state = end_pos

        # assign the initial location of agent
        self.start_state = start_pos
        self.agent_position = start_pos

        # respective actions of agents : up, down, left and right
        self.action_space = spaces.Discrete(4)
        self.holes = [[2, 5], [5, 2], [8, 5], [5, 8]]

        # set the observation space to (1,) to represent agent position in the grid world
        # staring from [0,size*size)
        self.observation_space = spaces.Box(low=0, high=size*size, shape=(1,), dtype=np.uint8)

    def step(self, action):
        info = {} # additional information

        reward = 0

        row  = self.agent_position // self.size
        col  = self.agent_position % self.size
        assert row <= self.size and row >= 0 or col <= self.size and col >= 0, "agent out of maze"
        room = self.get_room_id(row, col)
        print(room)

        # assert row != self.size//2, "agent is in wall"
        # assert col != self.size//2, "agent is in wall"

        if action == self.UP:
            if row != 0 and row - 1 != self.size//2:
                self.agent_position -= self.size
            elif [row - 1, col] in self.holes:
                self.agent_position -= self.size * 2
                print("get into")
            else:
                reward = 0
        elif action == self.LEFT:
            if col != 0 and col - 1 != self.size//2:
                self.agent_position -= 1
            elif [row, col-1] in self.holes:
                self.agent_position -= 2
                print("get into")
            else:
                reward = 0
        elif action == self.DOWN:
            if row != self.size - 1 and row + 1 != self.size//2:
                self.agent_position += self.size
            elif [row + 1, col] in self.holes:
                self.agent_position += self.size * 2
                print("get into")
            else:
                reward = 0
        elif action == self.RIGHT:
            if col != self.size - 1 and col + 1 != self.size//2:
                self.agent_position += 1
            elif [row, col + 1] in self.holes:
                self.agent_position += 2
                print("get into")
            else:
                reward = 0
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        done   = bool(self.agent_position == self.end_state)

        # reward agent when it is in the terminal cell, else reward = 0
        reward = 100 if done else reward

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


class TaskGenerator():
    def __init__(self, env):
        self.env = env
        self.env_size = env.size

    def create_source_tasks(self, target_task, mode='') -> np.ndarray:
        """Return source tasks for given mode.
        Args:
            mode: ['adjacent', 'inroom']
            target_task: target task: (2, 2)

        Returns:
            source tasks
        """

        assert mode, "Please specify mode!"

        tasks_rc = self.create_task()

        # Extract start position only
        tasks_start_pos = tasks_rc[:, 0]

        # classify each tasks with 4 different rooms
        left_upper = []
        right_upper = []
        left_lower = []
        right_lower = []
        for i in range(tasks_rc.shape[0]):
            task = tasks_start_pos[i]
            r, c = task[0], task[1]
            room_id = self.env.get_room_id(r, c)

            if room_id == 0:
                left_upper.append(task)
            elif room_id == 1:
                right_upper.append(task)
            elif room_id == 2:
                left_lower.append(task)
            elif room_id == 3:
                right_lower.append(task)

        tasks_divided = np.stack([left_upper, right_upper, left_lower, right_lower], axis=0)
        print(tasks_divided.shape)

        # possible end position on each room.
        # This type of source task creates 8 different tasks which terminate when agent can move onto adjacent room.
        end_pos_arr = np.array([
            [[2, 6],
             [6, 2]],
            [[2, 4],
             [6, 8]],
            [[4, 2],
             [8, 6]],
            [[8, 4],
             [4, 8]],
        ])

        # -> (N_room, N_tasks, 1, pos)
        # -> (N_room, N_tasks * 2, 1, pos)
        tasks_room = np.expand_dims(tasks_divided, axis=2)
        tasks_room = np.repeat(tasks_room, 2, axis=1)
        print(tasks_room.shape)

        # -> (N_room, N_tasks * 2, 1, pos)

        end_pos_arr = np.repeat(np.expand_dims(end_pos_arr, axis=2), tasks_divided.shape[1], axis=1)
        print(tasks_room.shape, end_pos_arr.shape)
        # -> (N_room, N_tasks * 2, 2, pos)
        tasks_room = np.concatenate([tasks_room, end_pos_arr], axis=2)
        # -> (N_room * N_tasks * 2, 2, pos)
        tasks = np.reshape(tasks_room, [tasks_room.shape[0] * tasks_room.shape[1], *tasks_room.shape[2:]])


        # create source task which start with room where target task's end position exists.
        # tasks_inroom: -> (N_tasks * 2, 2, pos)
        room_id = self.env.get_room_id(target_task[1][0], target_task[1][1])
        tasks_room[room_id, :, 1, :] = target_task[1]
        tasks_inroom = tasks_room[room_id]

        if mode == 'adjacent':
            return tasks
        elif mode == 'inroom':
            return tasks_inroom

    def create_task(self):
        start_pos = np.arange(0, self.env.size**2)
        end_pos = np.arange(0, self.env.size**2)

        # A set of possible target tasks
        xx, yy = np.meshgrid(start_pos, end_pos)
        tasks = np.array([xx.flatten(), yy.flatten()]).T

        # Delete tasks which has same start/end position.
        same_pos = np.concatenate([np.expand_dims(start_pos, 0), np.expand_dims(end_pos, 0)], axis=0).T
        idx_list = []
        for i in range(tasks.shape[0]):
            task = tasks[i]
            for j in range(same_pos.shape[0]):
                if np.array_equal(task, same_pos[j]):
                    idx_list.append(i)
        tasks = np.delete(tasks, idx_list, axis=0)

        # convert2(row, column) representations
        tasks_rc = np.zeros([tasks.shape[0], 2, 2])
        for i in range(tasks_rc.shape[0]):
            start_pos, end_pos = tasks[i][0], tasks[i][1]
            r_start, c_start = start_pos//self.env_size, start_pos%self.env_size
            r_end, c_end = end_pos//self.env_size, end_pos%self.env_size
            tasks_rc[i][0] = np.array([r_start, c_start])
            tasks_rc[i][1] = np.array([r_end, c_end])

        # Delete tasks which start/end at inside of wall
        idx_list = []
        for idx in range(tasks_rc.shape[0]):
            r_start, c_start = tasks_rc[idx][0]
            if r_start == self.env_size//2 or c_start == self.env_size//2:
                idx_list.append(idx)

            r_end, c_end = tasks_rc[idx][1]
            if r_end == self.env_size//2 or c_end == self.env_size//2:
                idx_list.append(idx)
        tasks_rc = np.delete(tasks_rc, idx_list, axis=0)

        return tasks_rc