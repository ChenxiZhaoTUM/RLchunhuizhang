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
        self.pi = self._softmax_cvt_theta_to_pi()
        self.eta = 0.1

    def _softmax_cvt_theta_to_pi(self, beta=1.):
        m, n = self.theta.shape  # m is state, n is action
        pi = np.zeros((m, n))  # policy pi
        exp_theta = np.exp(self.theta * beta)
        for r in range(m):
            pi[r, :] = exp_theta[r, :] / np.nansum(exp_theta[r, :])  # discrete probability distribution
        return np.nan_to_num(pi)

    def update_theta(self, s_a_history):
        T = len(s_a_history) - 1
        m, n = self.theta.shape
        delta_theta = self.theta.copy()
        for i in range(m):
            for j in range(n):
                if not (np.isnan(self.theta_0[i][j])):
                    s_a_i = [s_a for s_a in s_a_history if s_a[0] == i]
                    s_a_ij = [s_a for s_a in s_a_history if (s_a[0] == i and s_a[1] == j)]
                    N_i = len(s_a_i)
                    N_ij = len(s_a_ij)
                    delta_theta[i, j] = (N_ij - self.pi[i, j] * N_i) / T
        self.theta = self.theta + self.eta * delta_theta
        return self.theta

    def update_pi(self):
        self.pi = self._softmax_cvt_theta_to_pi()
        return self.pi

    def choose_action(self, state):
        action = np.random.choice(self.actions, p=self.pi[state, :])
        return action


agent = Agent()
done = False

stop_eps = 1e-4

while True:
    env = MazeEnv()
    state = env.reset()
    # trajectory
    s_a_history = [[state, np.nan]]
    state_history = [state]

    while True:
        action = agent.choose_action(state)
        s_a_history[-1][1] = action  # action of the last state
        state, reward, done, _ = env.step(action)
        s_a_history.append([state, np.nan])
        state_history.append(state)
        if state == 8 or done:
            break

    agent.update_theta(s_a_history)
    pi = agent.pi.copy()  # the last pi
    agent.update_pi()
    delta = np.sum(np.abs(agent.pi - pi))

    print(len(s_a_history), delta)
    if delta < stop_eps:
        break

print(s_a_history)
maze_display.play_process(state_history)
