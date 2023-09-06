import gym
import numpy as np
import maze_display


class MazeEnv(gym.Env):
    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:
            self.state -= 3
        elif action == 1:
            self.state += 1
        elif action == 2:
            self.state += 3
        elif action == 3:
            self.state -= 1

        done = False
        if self.state == 8:
            done = True
        return self.state, 1, done, {}  # state, reward, done, {}


class Agent:
    def __init__(self):
        self.actions = list(range(4))

        self.theta_0 = np.asarray([[np.nan, 1, 1, np.nan],  # s0
                      [np.nan, 1, np.nan, 1],  # s1
                      [np.nan, np.nan, 1, 1],  # s2
                      [1, np.nan, np.nan, np.nan],  # s3
                      [np.nan, 1, 1, np.nan],  # s4
                      [1, np.nan, np.nan, 1],  # s5
                      [np.nan, 1, np.nan, np.nan],  # s6
                      [1, 1, np.nan, 1]]  # s7
                     )

        self.theta = self.theta_0
        self.pi = self._cvt_theta_to_pi()

    # the most naive method to define pi
    def _cvt_theta_to_pi(self):
        m, n = self.theta.shape  # m is state, n is action
        pi = np.zeros((m, n))  # policy pi
        for r in range(m):
            pi[r, :] = self.theta[r, :] / np.nansum(self.theta[r, :])  # discrete probability distribution
        return np.nan_to_num(pi)

    def choose_action(self, state):
        action = np.random.choice(self.actions, p=self.pi[state, :])
        return action


env = MazeEnv()
state = env.reset()
agent = Agent()
done = False
state_history = [state]
action_history = []

while not done:
    action = agent.choose_action(state)
    state, reward, done, _ = env.step(action)

    action_history.append(action)
    state_history.append(state)

print(state_history)
maze_display.play_process(state_history)
