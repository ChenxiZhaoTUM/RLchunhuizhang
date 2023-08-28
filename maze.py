import numpy as np
from maze_display import play_process

###### agent action policy ######
# border & barrier
theta_0 = np.asarray([[np.nan, 1, 1, np.nan],  # s0
                      [np.nan, 1, np.nan, 1],  # s1
                      [np.nan, np.nan, 1, 1],  # s2
                      [1, np.nan, np.nan, np.nan],  # s3
                      [np.nan, 1, 1, np.nan],  # s4
                      [1, np.nan, np.nan, 1],  # s5
                      [np.nan, 1, np.nan, np.nan],  # s6
                      [1, 1, np.nan, 1]]  # s7
                     )


def cvt_theta_0_to_pi(theta):
    m, n = theta.shape  # m is state, n is action
    pi = np.zeros((m, n))  # policy pi

    for r in range(m):
        pi[r, :] = theta[r, :] / np.nansum(theta[r, :])  # discrete probability distribution

    return np.nan_to_num(pi)


pi = cvt_theta_0_to_pi(theta_0)


# print(pi)
# [[0.         0.5        0.5        0.        ]
#  [0.         0.5        0.         0.5       ]
#  [0.         0.         0.5        0.5       ]
#  [1.         0.         0.         0.        ]
#  [0.         0.5        0.5        0.        ]
#  [0.5        0.         0.         0.5       ]
#  [0.         1.         0.         0.        ]
#  [0.33333333 0.33333333 0.         0.33333333]]


def step(state, action):
    if action == 0:
        state -= 3
    elif action == 1:
        state += 1
    elif action == 2:
        state += 3
    elif action == 3:
        state -= 1
    return state


###### begin to play ######
state = 0
actions = list(range(4))  # actions = [0, 1, 2, 3]
state_history = [state]
action_history = []
while True:
    action = np.random.choice(actions, p=pi[state, :])  # at state, choose action by probability pi
    state = step(state, action)

    action_history.append(action)
    state_history.append(state)

    if state == 8:
        break

print(state_history)
print(action_history)

###### visualization ######
play_process(state_history)
