import time
import gym
from CartPole_display import display_frame_to_video
import os

if not os.path.exists('save'):
    os.makedirs('save')


env_name = 'CartPole-v0'
env = gym.make(env_name)

# one episode
state = env.reset()
done = False
total_reward = 0
steps = 0
frames = []

print(env.observation_space)

while not done:
    frames.append(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    print(f'action: {action}')
    observation, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1
    print(f'step: {steps}, state: {state}')
    time.sleep(0.2)

print(total_reward)

display_frame_to_video(frames, output='./save/rand_cartpole.gif')
display_frame_to_video(frames, output='./save/rand_cartpole.mp4')
