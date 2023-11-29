#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Khushal21csu188/RLLAB-SEM-5/blob/main/RL%20Exp-7.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import numpy as np

states = [0, 1, 2, 3, 4]
actions = [0, 1]
N_STATES = len(states)
N_ACTIONS = len(actions)
P = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # transition probability
R = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # rewards
P[0, 0, 1] = 1.0  # STATE ACTION and Reward
P[1, 1, 2] = 1.0
P[2, 0, 3] = 1.0
P[3, 1, 4] = 1.0
P[4, 0, 4] = 1.0

R[0, 0, 1] = 1
R[1, 1, 2] = 10
R[2, 0, 3] = 100
R[3, 1, 4] = 1000
R[4, 0, 4] = 1.0
gamma = 0.75

# initialize policy and value arbitrarily

policy = [0 for s in range(N_STATES)]
V = np.zeros(N_STATES)

print("Initial policy", policy)
# print V
# Initial policy [0, 0, 0, 0, 0]
is_value_changed = True
iterations = 0

while is_value_changed:
    is_value_changed = False
    iterations += 1
    # run value iteration for each state
    for s in range(N_STATES):
        q_best = V[s]
        for a in range(N_ACTIONS):
            q_sa = sum([P[s, a, s1] * (R[s, a, s1] + gamma * V[s1]) for s1 in range(N_STATES)])
            if q_sa > q_best:
                policy[s] = a
                q_best = q_sa
                is_value_changed = True
        V[s] = q_best

print("Iterations", iterations)
print("Final Policy")
print(policy)
print(V)


