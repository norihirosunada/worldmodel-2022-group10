import numpy as np

class TaskGenerator(object):
    """
    Generate task for this setting.
    """
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

        # -> (N_room, N_tasks * 2, 1, pos)

        end_pos_arr = np.repeat(np.expand_dims(end_pos_arr, axis=2), tasks_divided.shape[1], axis=1)
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