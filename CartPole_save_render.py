import time
import gym
from CartPole_display import display_frame_to_video

env_name = 'CartPole-v0'
env = gym.make(env_name)

state = env.reset()
done = False
total_reward = 0

frames = []
while not done:
    frames.append(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    print(action)
    observation, reward, done, info = env.step(action)
    total_reward += reward
    time.sleep(0.2)
env.close()
print(total_reward)
print(len(frames))
print(frames[0].shape)  # (400, 600, 3)

display_frame_to_video(frames)

