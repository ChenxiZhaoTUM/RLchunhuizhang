import random
import gym
import numpy as np
from CartPole_display import display_frame_to_video
import os
from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

if not os.path.exists('save'):
    os.makedirs('save')

### experience replay
# 构造批次化训练数据
# 让整个训练过程更稳定

# s_t, a_t => s_{t+1}
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# tr = Transition(1, 2, 3, 4)
# tr.state # 1

# trs = [Transition(1, 2, 3, 4), Transition(5, 6, 7, 8)]
# trs = Transition(*zip(*trs))
# trs  # Transition(state=(1, 5), action=(2, 6), next_state=(3, 7), reward=(4, 8))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            # placeholder
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity  # 使用模运算（%）来循环覆盖缓冲区的旧样本，确保缓冲区不会无限增长，而是保持固定容量 self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # input: batch_size * 4D (state)
    # output: batch_size * 2D (action)
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        return self.fc3(x2)


class Agent:
    def __init__(self, n_states, n_actions, eta=0.5, gamma=0.99, capacity=10000, batch_size=32):
        self.n_states = n_states
        self.n_actions = n_actions
        self.eta = eta
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = ReplayMemory(capacity)
        self.model = DQN(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        # list of transition
        batch = self.memory.sample(self.batch_size)
        # Transition, column: len(tuple) == batch_size
        batch = Transition(*zip(*batch))

        # s_t.shape: batch_size * 4
        state_batch = torch.cat(batch.state)
        # a_t.shape: batch_size * 1
        action_batch = torch.cat(batch.action)
        # r_{t+1}.shape: batch_size * 1
        reward_batch = torch.cat(batch.reward)
        # < batch_size
        non_final_next_state_batch = torch.cat([s for s in batch.next_state if s is not None])


        # 构造模型训练用的输入和输出（true）
        # s_t, input

        # pred: Q(s_t, a_t)
        # true: R_{t+1} + \gamma*\max_aQ(s_t, a)

        # 开启 eval 模式
        self.model.eval()

        # pred, batch_size*1
        state_action_values = self.model(state_batch).gather(dim=1, index=action_batch)

        # true: R_{t+1} + \gamma*\max_aQ(s_t, a)
        # tuple(map(lambda s: s is not None, batch.next_state)): batch_size 长度的 0/1
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(self.batch_size)
        # Q(s_{t+1}, a)
        next_state_values[non_final_mask] = self.model(non_final_next_state_batch).max(dim=1)[0].detach()

        # (batch_size, )
        expected_state_action_values = reward_batch + self.gamma * next_state_values


        # 开启train mode
        self.model.train()

        # expected_state_action_values.unsqueeze(1): (batch_size, ) => (batch_size, 1)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def update_q_function(self):
        self._replay()

    def choose_action(self, state, episode):
        eps = 0.5*1/(1+episode)

        if random.random() < eps:
            # explore
            action = torch.IntTensor([[random.randrange(self.n_actions)]])  # 将action转化成2D，与action_space维度相同
        else:
            # exploit
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(dim=1)[1].view(1, 1)  # self.model(state)计算状态的Q值估计，max(dim=1)选择具有最高Q值估计的动作的索引，view(1, 1)将索引变成一个2D张量以便与选择的动作格式一致
        return action


# interaction between agent and environment
env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n  # 2

agent = Agent(n_states, n_actions)

max_episodes = 500
max_steps = 200

continue_success_episodes = 0

for episode in range(max_episodes):
    state = env.reset()
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = state.unsqueeze(0)

    for step in range(max_steps):

        # action已被拓展为1*1的2D tensor
        action = agent.choose_action(state, episode)
        # transition on env
        next_state, _, done, _ = env.step(action.item())
        # 此时得到的next_state是一个包含四个数的一维array

        if done:
            next_state = None

            if step < 195:
                reward = torch.FloatTensor([-1.])  # 创建一个包含单个浮点数1.0的张量
                continue_success_episodes = 0
            else:
                reward = torch.FloatTensor([1.])
                continue_success_episodes += 1
        else:
            reward = torch.FloatTensor([0])
            # 将next_state从NumPy数组转换为PyTorch张量  =>(4, )
            next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
            # 将next_state张量的形状进行调整  (4, ) => (1, 4) 便于后续torch.cat
            next_state = next_state.unsqueeze(0)

        agent.memorize(state, action, next_state, reward)
        agent.update_q_function()
        state = next_state

        if done:
            print(f'episode: {episode}, steps: {step}')
            break

    if continue_success_episodes >= 10:
        print(f'continue success (step > 195) more than 10 times')
        break
