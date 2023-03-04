import numpy as np
import env
from student import QLearningAgent


NB_EPISODE = 1000    # エピソード数
EPSILON = .2    # 探索率
ALPHA = .3      # 学習率
GAMMA = .80     # 割引率
ACTIONS = np.arange(4)  # 行動の集合
MAX_STEP = 10

task = [0, 88]
env = env.make_env(*task)
init_state = np.array([task[0]]).astype(np.uint8)

# エージェントの初期化
agent = QLearningAgent(
  env=env,
  alpha=ALPHA,
  gamma=GAMMA,
  epsilon=EPSILON,  # 探索率
  actions=ACTIONS,   # 行動の集合
  observation=init_state
)  # Q学習エージェント
rewards = []    # 評価用報酬の保存
is_end_episode = False  # エージェントがゴールしてるかどうか？

episode_reward = []  # 1エピソードの累積報酬

episode_count = 0
while episode_count <= NB_EPISODE:
    while(is_end_episode == False):    # ゴールするまで続ける
        action = agent.act()  # 行動選択
        state, reward, is_end_episode = env.step(action)[0:3]
        agent.observe(state, reward)   # 状態と報酬の観測
        episode_reward.append(reward)
        env.render()
        print()

    episode_count += 1
print(agent.reward_history)
print(agent.q_values)
print(episode_count)
print(agent.dict_to_table(agent.q_values))
