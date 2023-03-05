import numpy as np
import env
from student import QLearningAgent
import pandas as pd


NB_EPISODE = 10    # エピソード数
EPSILON = .1
# 探索率
ALPHA = .2      # 学習率
GAMMA = 0.9     # 割引率
ACTIONS = np.arange(4)  # 行動の集合
MAX_STEP = 1000

target_task = [0, 88]

maze = env.make_env()
task_gen = env.TaskGenerator(maze)
source_tasks = task_gen.create_source_tasks(maze.convert_task_to_pos(target_task), mode='adjacent')

start_pos, end_pos = source_tasks[3][0], source_tasks[3][1]
maze.set_task(start_pos, end_pos)
init_state = np.array([maze.convert_pos_to_idx(start_pos)]).astype(np.uint8)

man_dist =  maze.manhattan_distance(start_pos, end_pos)  # manhattan distance between start and goal state.

# エージェントの初期化
agent = QLearningAgent(
  env=maze,
  alpha=ALPHA,
  gamma=GAMMA,
  epsilon=EPSILON,  # 探索率
  actions=ACTIONS,   # 行動の集合
  observation=init_state
)  # Q学習エージェント
maze.reset()
rewards = []    # 評価用報酬の保存
is_end_episode = False  # エージェントがゴールしてるかどうか？
delta = 1  # エージェントがタスクを完了したかを判定する際に用いるハイパーパラメータ

episode_reward = []  # 1エピソードの累積報酬

episode_count = 0
while episode_count < NB_EPISODE:
    num_staps = 0
    while not is_end_episode:    # ゴールするまで続ける
        action = agent.act()  # 行動選択
        state, reward, is_end_episode = maze.step(action)[0:3]
        agent.observe(state, reward)   # 状態と報酬の観測
        episode_reward.append(reward)
        maze.render()
        print()
        num_staps += 1
        if num_staps > 1000:
            break
    agent.dict_to_table(agent.q_values)
    maze.reset()
    is_end_episode = False
    episode_count += 1
    if num_staps < man_dist + delta:
        #pass
        break
print(episode_count)
print(pd.DataFrame(agent.state_values))
