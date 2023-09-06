import random
import gym
import matplotlib.pyplot as plt
import numpy as np
from CartPole_display import display_frame_to_video
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

if not os.path.exists('save'):
    os.makedirs('save')


class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_layers, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_states, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.softmax(self.fc2(x1), dim=1)
        return x2

    def choose_action(self, state):
        # (4, ) => (1, 4)
        state = torch.from_numpy(state).float().unsqueeze(0)

        # 计算概率
        probs = self.forward(state)

        # 根据概率采样
        highest_prob_action = np.random.choice(self.n_actions, p=np.squeeze(probs.detach().numpy()))
        # np.squeeze(probs.detach().numpy()) 将PyTorch张量转换为NumPy数组，并去除多余的维度，以获得一维数组表示动作概率
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


GAMMA = 0.9


# Gt = sum_{k=0 to infinity}((GAMMA**k)*R_{t+k+1})

# 由一次episode的trajectory产生的rewards
def discounted_future_rewards(rewards):
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        # 累加t时刻开始后的reward
        for r in rewards[t:]:
            Gt += (GAMMA ** pw) * r
            pw += 1
        discounted_rewards.append(Gt)
    return discounted_rewards


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = discounted_future_rewards(rewards)

    # 将discounted_reward归一化
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_grads = []
    for log_probs, Gt in zip(log_probs, discounted_rewards):
        policy_grads.append(-log_probs * Gt)

    policy_network.optimizer.zero_grad()
    policy_grads = torch.stack(policy_grads).sum()
    policy_grads.backward()
    policy_network.optimizer.step()


env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n  # 2
policy_net = PolicyNetwork(n_states, n_actions, 128)

max_episodes = 3000
max_steps = 200

num_steps = []
avg_num_steps = []
all_rewards = []

for episode in range(max_episodes):
    state = env.reset()
    log_probs = []
    rewards = []
    for step in range(max_steps):
        action, log_prob = policy_net.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            # 完成一次episode，得到一次完整的trajectory
            update_policy(policy_net, rewards, log_probs)
            num_steps.append(step)
            avg_num_steps.append(np.mean(num_steps[-10:]))  # 取最近的10次steps进行平均
            all_rewards.append(sum(rewards))
            if episode % 50 == 0:
                print(f'episodes: {episode}, total reward: {sum(rewards)}, average_reward: {np.mean(all_rewards)}, length: {step}')
                # 当前episode的total reward和运行步数step
            break

        state = next_state

plt.plot(num_steps)
plt.plot(avg_num_steps)
plt.xlabel('episode')
plt.show()
