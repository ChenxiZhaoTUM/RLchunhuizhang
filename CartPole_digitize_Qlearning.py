import random
import gym
import numpy as np
from CartPole_display import display_frame_to_video
import os

if not os.path.exists('save'):
    os.makedirs('save')


class Agent:
    def __init__(self, n_states, action_space, eta=0.5, gamma=0.99, NUM_DIGITIZED=6):
        self.action_space = action_space
        self.eta = eta
        self.gamma = gamma
        self.NUM_DIGITIZED = NUM_DIGITIZED
        self.q_table = np.random.uniform(0, 1, size=(
            NUM_DIGITIZED ** n_states, self.action_space.n))  # 行数：状态离散化后的可能状态数量；列数：可选的动作数量

    # 分桶
    @staticmethod
    def _bins(clip_min, clip_max, num_bins):
        return np.linspace(clip_min, clip_max, num_bins + 1)[1: -1]

    # 将四位6进制数映射为id （数字乘以NUM_DIGITIZED ** n_states的加和）
    @staticmethod
    def _digitize_state(observation, NUM_DIGITIZED):
        pos, cart_v, angle, pole_v = observation
        digitized = [np.digitize(pos, bins=Agent._bins(-4.8, 4.8, NUM_DIGITIZED)),
                     np.digitize(cart_v, bins=Agent._bins(-3., 3, NUM_DIGITIZED)),
                     np.digitize(angle, bins=Agent._bins(-0.418, 0.418, NUM_DIGITIZED)),
                     np.digitize(pole_v, bins=Agent._bins(-2, 2, NUM_DIGITIZED))]

        ind = sum(
            [d * (NUM_DIGITIZED ** i) for i, d in enumerate(digitized)])
        return ind

    def choose_action(self, state, episode):
        eps = 0.5 * 1 / (episode + 1)
        state_ind = Agent._digitize_state(state, self.NUM_DIGITIZED)
        # epsilon greedy
        if random.random() < eps:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.q_table[state_ind, :])
        return action

    def q_learning_digitize(self, obs, action, reward, obs_next):
        obs_ind = Agent._digitize_state(obs, self.NUM_DIGITIZED)
        obs_next_ind = Agent._digitize_state(obs_next, self.NUM_DIGITIZED)

        self.q_table[obs_ind, action] = self.q_table[obs_ind, action] + self.eta * (
                reward + self.gamma * np.nanmax(self.q_table[obs_next_ind, :]) - self.q_table[obs_ind, action])


env_name = 'CartPole-v0'
env = gym.make(env_name)
env.reset()
n_states = env.observation_space.shape[0]  # obeservation_space的行数
action_space = env.action_space

agent = Agent(n_states, action_space)

max_episodes = 1000
max_steps = 200

continue_success_episodes = 0
learning_finish_flag = False
frames = []

for episode in range(max_episodes):
    obs = env.reset()

    for step in range(max_steps):
        if learning_finish_flag:
            frames.append((env.render(mode='rgb_array')))

        action = agent.choose_action(obs, episode)
        obs_next, _, done, _ = env.step(action)

        if done:
            if step < 195:
                reward = -1
                continue_success_episodes = 0
            else:
                reward = 1
                continue_success_episodes += 1
        else:
            reward = 0

        agent.q_learning_digitize(obs, action, reward, obs_next)
        obs = obs_next

        if done:
            print(f'episode: {episode}, finish after {step} time steps')
            break

    if learning_finish_flag:
        break

    if continue_success_episodes >= 10:
        learning_finish_flag = True
        print(f'continue success (step > 195) more than 10 times')

if learning_finish_flag:
    display_frame_to_video(frames, output='./save/cartpole_qlearning.mp4')
