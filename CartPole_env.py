import time
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)

state = env.reset()
done = False
total_reward = 0

while not done:
    env.render(mode='human')
    action = env.action_space.sample()
    print(action)
    observation, reward, done, info = env.step(action)
    total_reward += reward
    time.sleep(0.2)
print(total_reward)
