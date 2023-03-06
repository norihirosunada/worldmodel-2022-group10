import numpy as np
from student import QLearningAgent
from task_generator import TaskGenerator
import utils
from collections import deque

import torch
from torch.nn import functional as F

from teacher import Teacher
from env import GridWorld


class StudentTrainer(object):
    def __init__(self,
                 env,
                 target_task,
                 n_episodes,
                 epsilon,
                 alpha,
                 gamma,
                 actions,
                 max_step,
                 ):
        self.env = env
        self.target_task = target_task
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.max_step = max_step

        self.student = None

    def reset_student(self, init_state):
        return QLearningAgent(
            env=self.env,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            actions=self.actions,
            observation=init_state,
        )

    def step(self, start_pos: tuple, end_pos: tuple, is_transfer=True, training=True) -> (int, np.ndarray, bool):
        """Training student at given source task until converged.
        Args:
            start_pos: tuple of start position (row, col)
            end_pos: tuple of end position
            is_transfer: whether to transfer student's q-values.
            training: whether to train student
        Returns:
            episode_count: number of episodes student have taken.
            value_table: student's state-value table at last step
            done: Whether student converged on target task.
            student_q_values: student's q-value
        """

        # Initialize environment
        self.env.set_task(start_pos, end_pos)
        init_state = np.array([self.env.convert_pos_to_idx(start_pos)]).astype(np.uint8)
        self.env.reset()

        # Set condition of convergence
        man_dist = self.env.manhattan_distance(start_pos, end_pos)

        # Initialize student
        if not is_transfer:
            self.student = self.reset_student(init_state)

        done = False
        task = np.concatenate([start_pos, end_pos], axis=0)

        is_end_episode = False
        delta = 1

        episode_reward = []
        episode_count = 0

        # Learning student until converged.
        while episode_count < self.n_episodes:
            num_staps = 0
            while not is_end_episode:
                action = self.student.act()
                state, reward, is_end_episode = self.env.step(action)[0:3]
                self.student.observe(state, reward, training=training)
                episode_reward.append(reward)
                if num_staps > 1000:
                    break
                num_staps += 1
            self.student.dict_to_table(self.student.q_values)
            self.env.reset()
            is_end_episode = False
            episode_count += 1

            # If number of actions have student taken this episode is lower than this condition,
            # reckon student converged on this task.
            if num_staps < man_dist + delta:
                if (self.target_task[0].numpy() == task).all():
                    done = True
                break

        student_state = torch.from_numpy(self.student.state_values.astype(np.float32)).view(1, 1, *self.student.state_values.shape)
        
        return episode_count, student_state,  done


class TeacherTrainer(object):
    def __init__(self,
                 env,
                 gamma=0.9,
                 lambda_=0.9,
                 ):
        """
        Args:
            teacher:
            env:
            gamma:
            lambda_:
        """
        self.student_env = env
        self.gamma = gamma
        self.lambda_ = lambda_

        self.teacher = None

    def train_teacher(self, target_task: np.ndarray, cmdp_episodes: int, n_step: int, max_step):
        """Training teacher's curriculum creation.
        Args:
            target_task: (2, 2) -> [[r_start, c_start], [r_end, c_end]]
            cmdp_episodes: number of curriculum MDP episodes
            n_step: number of steps when calculate lambda target value
            max_step: upper limit for action steps
        Returns:

        """

        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create source tasks for given target task.
        task_gen = TaskGenerator(self.student_env)
        source_adj = task_gen.create_source_tasks(target_task, mode='adjacent')
        source_inr = task_gen.create_source_tasks(target_task, mode='inroom')
        source_tasks = np.concatenate([source_adj, source_inr], axis=0)
        target_task_arr = target_task
        target_task = torch.unsqueeze(
            torch.from_numpy(target_task.reshape(4).astype(np.float32)).to(device), dim=0)
        
        # Initialize student trainer for given target task.
        student_trainer = StudentTrainer(
            self.student_env,
            target_task,
            n_episodes=100,
            epsilon=0.1,
            alpha=0.1,
            gamma=0.9,
            actions=np.arange(4),
            max_step=1000
        )
        
        # Initializer teacher for given target task
        self.teacher = Teacher(
            state_shape=(1, 1, self.student_env.size, self.student_env.size),
            task_dim=len(source_tasks)
        )

        # list of dict -> {'state', 'reward', 'action'}
        experiences = []

        # Transport model to device
        self.teacher.value_model.to(device)
        self.teacher.action_model.to(device)

        # Optimizer settings
        value_lr = 1e-4
        action_lr = 1e-4
        epsilon = 1e-6
        clip_grad_norm = 100

        value_optimizer = torch.optim.Adam(self.teacher.value_model.parameters(), lr=value_lr, eps=epsilon)
        action_optimizer = torch.optim.Adam(self.teacher.action_model.parameters(), lr=action_lr, eps=epsilon)
        
        for episode in range(cmdp_episodes):

            student_state = torch.zeros(1, 1, self.student_env.size, self.student_env.size)

            step_count = 0
            done = False
            is_transfer = False

            while not done:
                experience = {}

                # make for teacher's 1 step
                state_embedd = self.teacher.encoder(student_state)
                state_embedd = state_embedd.view(state_embedd.shape[:2])
                task_prob = self.teacher.action_model(state_embedd, target_task)
                task = source_tasks[torch.argmax(task_prob)]

                # run at source task
                episode_count, student_state, done = student_trainer.step(task[0], task[1], is_transfer)

                # run at target task
                # episode_count, _, done = student_trainer.step(target_task_arr[0], target_task_arr[1], is_transfer, False)

                is_transfer = True
                experience['action'] = task
                experience['reward'] = -1 * episode_count
                experience['state_embedd'] = state_embedd
                experiences.append(experience)

                if len(experiences) >= n_step:
                    # flatten target task
                    lambda_target_value = self.td_error(experiences, target_task)
                    estimated_value = self.teacher.value_model(experiences[0]['state_embedd'].detach(), target_task)

                    # Update action model
                    action_loss = estimated_value
                    action_optimizer.zero_grad()
                    action_loss.backward(retain_graph=True)
                    utils.clip_grad_norm_(self.teacher.action_model.parameters(), clip_grad_norm)
                    action_optimizer.step()

                    # Update Value model
                    value_loss = 0.5 * F.mse_loss(estimated_value, lambda_target_value)
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    utils.clip_grad_norm_(self.teacher.value_model.parameters(), clip_grad_norm)
                    value_optimizer.step()

                    print(f'Action loss: {action_loss.item():.2f} Value loss: {value_loss.item():.2f}')

                    experiences = []
                step_count += 1
                # Terminate when reached max action steps
                if max_step <= step_count:
                    break

    def td_error(self, experiences, target_task):

        # Estimate state values from n_step experiences
        target_task = target_task.repeat(len(experiences), 1)
        state_values = self.teacher.value_model(torch.squeeze(torch.stack([e['state_embedd'] for e in experiences], dim=0), 1), target_task)
        state_values = state_values[:, 0]

        rewards = torch.unsqueeze(torch.FloatTensor([e['reward'] for e in experiences]), dim=1)
        lambda_target_value = utils.lambda_target(rewards, state_values, self.gamma, self.lambda_)[0, 0]

        return lambda_target_value


if __name__ == '__main__':

    student_env = GridWorld()
    # target task creation
    task_gen = TaskGenerator(student_env)
    target_tasks = task_gen.create_task()
    idx = np.random.randint(0, len(target_tasks))
    target_task = target_tasks[idx]

    teacher_trainer = TeacherTrainer(
        env=student_env,
    )

    teacher_trainer.train_teacher(
        target_task=target_task,
        cmdp_episodes=100,
        n_step=10,
        max_step=1000
    )
