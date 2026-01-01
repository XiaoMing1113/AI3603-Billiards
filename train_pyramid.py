import pooltool as pt
from agent import (
    PyramidAgent,
    BasicAgent,
    BayesianBasicAgent,
    analyze_shot_for_reward,
    analyze_shot_for_reward_agent,
)
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import copy

# 超参数
BATCH_SIZE = 64
GAMMA = 0.99
LR = 3e-4
MAX_EPISODES = 1000
MEMORY_CAPACITY = 10000
UPDATE_FREQ = 100  # 每多少步更新一次网络
TARGET_UPDATE_FREQ = 10 # 每多少个episode更新一次目标网络

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, reward, next_state, done = map(np.stack, zip(*batch))
        return state, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def train():
    # 1. 初始化
    log_dir = f"runs/Pyramid_Train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    agent = PyramidAgent() # 包含 ValueNetwork
    opponent_agent = BayesianBasicAgent()  # 对手：贝叶斯 Agent
    buffer = ReplayBuffer(MEMORY_CAPACITY)
        
    # 初始化球桌 (标准8球)
    table = pt.Table.default()
    gametype = pt.GameType.EIGHTBALL
    # 注意：这里直接使用 pooltool 的底层接口，不再使用 get_ruleset
    # 因为 get_ruleset 返回的对象在不同版本中可能不同
    # 我们主要利用 PoolEnv 中的逻辑，但 PoolEnv 耦合了 Agent 调用
    #
    # 为了简化，我们在这里直接实例化一个 poolenv.PoolEnv 用于训练
    # 这样可以复用 poolenv.py 中已经调试好的规则逻辑

    from poolenv import PoolEnv
    env = PoolEnv()

    global_steps = 0

    print("开始训练 Pyramid Agent (Value Network)...")

    for i_episode in range(MAX_EPISODES):
        # PoolEnv.reset 需要 target_ball 参数
        env.reset(target_ball='solid') # 默认训练时 A 打 solid

        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            active_player = env.get_curr_player() # 'A' or 'B'

            # 简化：假设 Agent 总是 Player A
            if active_player == 'A':
                # 1. 获取当前状态 (固定用 A 视角，避免回合切换导致 targets 语义变化)
                balls, my_targets, table = env.get_observation('A')

                # 处理状态特征 (s)
                state_tensor = agent.process_state(balls, my_targets, table)
                state_np = state_tensor.cpu().numpy() # 存入 buffer 用 numpy

                # 2. 决策
                action = agent.decision(balls, my_targets, table)

                # 3. 执行
                # PoolEnv.take_shot 负责物理模拟和规则判定
                # 但我们需要在 take_shot 内部获取 shot 对象来计算 dense reward
                # PoolEnv.take_shot 会更新 self.balls 和 self.last_state

                # 记录击球前的快照用于计算 Reward (PoolEnv 内部也有 last_state)
                last_state_snapshot = {bid: copy.deepcopy(b) for bid, b in balls.items()}

                # 执行动作
                step_info = env.take_shot(action)

                # 4. 计算奖励 (r)
                # 我们需要访问 env.shot_record[-1] 来获取最近一次的 shot 系统
                if len(env.shot_record) > 0:
                    last_shot = env.shot_record[-1]
                    reward = analyze_shot_for_reward_agent(last_shot, last_state_snapshot, my_targets)
                else:
                    reward = 0

                episode_reward += reward

                # 5. 获取下一状态 (s')
                next_balls, next_targets, _ = env.get_observation('A')
                next_state_tensor = agent.process_state(next_balls, next_targets, table)
                next_state_np = next_state_tensor.cpu().numpy()

                # 判断是否结束 (本局结束)
                done, info = env.get_done()

                # 6. 存入 Buffer
                buffer.push(state_np, reward, next_state_np, done)

                global_steps += 1
                episode_steps += 1

                # 7. 更新网络
                if len(buffer) > BATCH_SIZE and global_steps % UPDATE_FREQ == 0:
                    b_s, b_r, b_ns, b_d = buffer.sample(BATCH_SIZE)

                    b_s = torch.FloatTensor(b_s).to(agent.device)
                    b_r = torch.FloatTensor(b_r).unsqueeze(1).to(agent.device)
                    b_ns = torch.FloatTensor(b_ns).to(agent.device)
                    b_d = torch.FloatTensor(b_d).unsqueeze(1).to(agent.device)

                    # 计算 Target Value
                    with torch.no_grad():
                        target_v = agent.target_value_net(b_ns)
                        target_val = b_r + GAMMA * target_v * (1 - b_d)

                    # 计算 Current Value
                    current_val = agent.value_net(b_s)

                    # Loss
                    loss = F.mse_loss(current_val, target_val)

                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()

                    writer.add_scalar("Train/Loss", loss.item(), global_steps)
                    writer.add_scalar("Train/AvgValue", current_val.mean().item(), global_steps)

            else:
                # 对手回合 (Player B) - 使用 BasicAgent
                balls, my_targets, table = env.get_observation()
                action = opponent_agent.decision(balls, my_targets, table)
                env.take_shot(action)
                done, info = env.get_done()

        # Episode 结束
        print(f"Episode {i_episode+1}: Reward={episode_reward:.1f}, Steps={episode_steps}")
        writer.add_scalar("Train/Episode_Reward", episode_reward, i_episode)

        # 更新 Target Network
        if (i_episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.target_value_net.load_state_dict(agent.value_net.state_dict())

        # 保存模型
        if (i_episode + 1) % 10 == 0:
            torch.save(agent.value_net.state_dict(), "value_net.pth")
            print("模型已保存: value_net.pth")

if __name__ == "__main__":
    train()
