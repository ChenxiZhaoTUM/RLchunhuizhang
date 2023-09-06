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
        reward = 0
        if self.state == 8:
            done = True
            reward = 1
        return self.state, reward, done, {}


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
        self.Q = np.random.rand(*self.theta_0.shape) * self.theta_0
        self.eta = 0.1
        self.gamma = 0.9
        self.eps = 0.5

    def _cvt_theta_to_pi(self):
        m, n = self.theta.shape  # m is state, n is action
        pi = np.zeros((m, n))  # policy pi
        for r in range(m):
            pi[r, :] = self.theta[r, :] / np.nansum(self.theta[r, :])  # discrete probability distribution
        return np.nan_to_num(pi)

    def get_action(self, s):
        if np.random.rand() < self.eps:
            action = np.random.choice(self.actions, p=self.pi[s, :])  # randomly choose by eps
        else:
            action = np.nanargmax(self.Q[s, :])  # according to the maximum value
        return action

    def sarsa(self, s, a, r, s_next, a_next):  # sarsa: s_next and a_next are determined
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])

    def q_learning(self, s, a, r, s_next):
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * np.nanmax(self.Q[s_next, :]) - self.Q[s, a])


env = MazeEnv()
agent = Agent()
done = False
epoch = 0

while True:
    old_Q = np.nanmax(agent.Q, axis=1)
    s = env.reset()
    a = agent.get_action(s)
    # trajectory
    s_a_history = [[s, np.nan]]
    state_history = [s]

    while True:
        s_a_history[-1][1] = a  # action of the last state of this moment

        s_next, reward, done, _ = env.step(a)  # next time
        s_a_history.append([s_next, np.nan])
        state_history.append(s_next)
        if done:
            a_next = np.nan
        else:
            a_next = agent.get_action(s_next)
        # agent.sarsa(s, a, reward, s_next, a_next)  # update Sarsa
        agent.q_learning(s, a, reward, s_next)  # update Q_learning
        if done:
            break
        else:
            s = env.state
            a = a_next

    update = np.sum(np.abs(np.nanmax(agent.Q, axis=1) - old_Q))  # update amplitude
    epoch += 1
    agent.eps /= 2

    print(epoch, update, len(s_a_history))
    if epoch > 100 or update < 1e-5:
        break


print(s_a_history)
print(agent.Q)
maze_display.play_process(state_history)
