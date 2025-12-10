import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from poolenv import PoolEnv
from agent import NewAgent, analyze_shot_for_dense_reward

# Hyperparameters
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
BATCH_SIZE = 256
BUFFER_SIZE = 100000
UPDATES_PER_STEP = 1
START_STEPS = 1000  # Random steps before training
MAX_EPISODES = 5000
SAVE_INTERVAL = 100

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        next_state = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state
        # action is already numpy array usually
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.stack(action), np.array(reward), np.stack(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class SACTrainer:
    def __init__(self):
        self.env = PoolEnv()
        self.agent = NewAgent(model_path="sac_model.pth")
        
        self.agent.actor.train()
        self.agent.critic.train()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.agent.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.agent.critic.parameters(), lr=LR)
        
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.start_episode = 1
        
        # Automatic Entropy Tuning (Optional, keeping alpha fixed for simplicity first, or implement if needed)
        self.target_entropy = -self.agent.action_dim
        # Initialize log_alpha to match ALPHA
        self.log_alpha = torch.tensor([np.log(ALPHA)], requires_grad=True, device=self.agent.device, dtype=torch.float32)
        self.alpha = self.log_alpha.exp().detach() # Initialize alpha consistent with log_alpha
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)

        # TensorBoard
        self.writer = SummaryWriter(log_dir="runs/SAC_Experiment")

    def select_action(self, balls, my_targets, table, evaluate=False):
        """选择动作"""
        # Get tensor state
        state_tensor = self.agent.process_state(balls, my_targets, table)
        
        if len(self.memory) < START_STEPS and not evaluate:
             # Random action in [-1, 1] space for exploration during warm-up
             action_norm = np.random.uniform(-1, 1, size=self.agent.action_dim)
        else:
            with torch.no_grad():
                if evaluate:
                    _, _, mean = self.agent.actor.sample(state_tensor.unsqueeze(0))
                    action_norm = mean.cpu().numpy()[0]
                else:
                    action, _, _ = self.agent.actor.sample(state_tensor.unsqueeze(0))
                    action_norm = action.cpu().numpy()[0]
                    
        return action_norm, state_tensor.cpu().numpy()

    def update_parameters(self, batch_size):
        if len(self.memory) < batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.agent.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.agent.device)
        action_batch = torch.FloatTensor(action_batch).to(self.agent.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.agent.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.agent.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.agent.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.agent.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * GAMMA * min_qf_next_target

        qf1, qf2 = self.agent.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.agent.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.agent.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Alpha Tuning
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        
        # Soft update
        for target_param, param in zip(self.agent.critic_target.parameters(), self.agent.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha.item()

    def train(self):
        print("开始训练 SAC Agent...")
        
        total_steps = 0
        global_update_steps = 0
        
        for i_episode in range(self.start_episode, MAX_EPISODES + 1):
            # Reset Env
            target_type = np.random.choice(['solid', 'stripe'])
            self.env.reset(target_ball=target_type)
            
            ep_reward = 0
            ep_steps = 0
            done = False
            
            while not done:
                total_steps += 1
                ep_steps += 1
                curr_player = self.env.get_curr_player() # 'A' or 'B'
                
                # Observation
                balls, my_targets, table = self.env.get_observation()
                
                if curr_player == 'A':
                    # Select action
                    action_norm, state_vec = self.select_action(balls, my_targets, table)
                    # Decode to physical action
                    phys_action = self.agent.decode_action(action_norm)
                else:
                    # Opponent (Random)
                    phys_action = self.agent._random_action()
                
                # Record state before shot
                last_state = copy.deepcopy(self.env.balls)
                targets_before = copy.deepcopy(my_targets)
                
                # Execute
                self.env.take_shot(phys_action)
                
                # Get Result
                last_shot = self.env.shot_record[-1]
                done, info = self.env.get_done()
                
                if curr_player == 'A':
                    # Calculate Dense Reward
                    reward = analyze_shot_for_dense_reward(last_shot, last_state, targets_before)
                    
                    # Next state
                    # Note: We need to process next state. But if done, next state might not matter as much (masked by done)
                    # We get next observation
                    next_balls, next_targets, next_table = self.env.get_observation()
                    next_state_vec = self.agent.process_state(next_balls, next_targets, next_table).cpu().numpy()
                    
                    # Store in replay buffer
                    self.memory.push(state_vec, action_norm, reward, next_state_vec, done)
                    
                    ep_reward += reward
                    
                    # Update parameters
                    if len(self.memory) > BATCH_SIZE:
                        for _ in range(UPDATES_PER_STEP):
                            qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_val = self.update_parameters(BATCH_SIZE)
                            global_update_steps += 1
                            
                            # Log training metrics
                            if global_update_steps % 100 == 0:
                                self.writer.add_scalar("Loss/Critic1", qf1_loss, global_update_steps)
                                self.writer.add_scalar("Loss/Critic2", qf2_loss, global_update_steps)
                                self.writer.add_scalar("Loss/Actor", policy_loss, global_update_steps)
                                self.writer.add_scalar("Loss/Alpha", alpha_loss, global_update_steps)
                                self.writer.add_scalar("Values/Alpha", alpha_val, global_update_steps)
                
                if done:
                    break
            
            alpha_val = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
            print(f"Episode {i_episode}\t Reward: {ep_reward:.2f}\t Steps: {ep_steps}\t Total Steps: {total_steps}\t Alpha: {alpha_val:.4f}")
            self.writer.add_scalar("Rollout/Episode_Reward", ep_reward, i_episode)
            self.writer.add_scalar("Rollout/Episode_Length", ep_steps, i_episode)
            
            if i_episode % SAVE_INTERVAL == 0:
                checkpoint = {
                    'actor': self.agent.actor.state_dict(),
                    'critic': self.agent.critic.state_dict(),
                    'log_alpha': self.log_alpha
                }
                torch.save(checkpoint, "sac_model.pth")
                print("模型已保存: sac_model.pth")

if __name__ == '__main__':
    trainer = SACTrainer()
    trainer.train()
