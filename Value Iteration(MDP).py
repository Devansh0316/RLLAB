#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Khushal21csu188/RLLAB-SEM-5/blob/main/RL%20Exp-8.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import numpy as np

def value_iteration(states, actions, transitions, rewards, gamma=0.9, epsilon=1e-6, max_iterations=1000):
    num_states = len(states)
    num_actions = len(actions)

    V = np.zeros(num_states)

    for _ in range(max_iterations):
        prev_V = np.copy(V)

        for s in range(num_states):
            Q_values = [sum(transitions[s, a, s_prime] * (rewards[s, a, s_prime] + gamma * V[s_prime])
                            for s_prime in range(num_states)) for a in range(num_actions)]

            V[s] = max(Q_values)

        if np.max(np.abs(V - prev_V)) < epsilon:
            break

    policy = [np.argmax([sum(transitions[s, a, s_prime] * (rewards[s, a, s_prime] + gamma * V[s_prime])
                              for s_prime in range(num_states)) for a in range(num_actions)])
              for s in range(num_states)]

    return V, policy

# Example usage:
states = [0, 1, 2, 3]
actions = [0, 1]
transitions = np.array([[[0.5, 0.5, 0, 0], [0.7, 0.3, 0, 0]],
                        [[0, 0.8, 0.2, 0], [0, 0, 1, 0]],
                        [[0, 0, 0.4, 0.6], [0, 0, 0, 1]],
                        [[0, 0, 0, 1], [0, 0, 0, 1]]])
rewards = np.array([[[1, -1, 0, 0], [2, 0, 0, 0]],
                    [[0, -1, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 5, -1], [0, 0, 0, -1]],
                    [[0, 0, 0, 10], [0, 0, 0, 10]]])

optimal_value_function, optimal_policy = value_iteration(states, actions, transitions, rewards)
print("Optimal Value Function:", optimal_value_function)
print("Optimal Policy:", optimal_policy)

