#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/aryansuri42/Reinforcement-learning-21csu467/blob/main/First_Visit_Every_Visit_Monte_Carlo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ##**First Visit**

# In[1]:


import numpy as np

# Define the episode
episodes = [
    [('A', 3), ('A', 2), ('B', -4), ('A', 4), ('B', -3)],
    [('B', -2), ('A', 3), ('B', -3)]
    ]

# Define the states
states = ['A', 'B']

# Initialize the value function
V = {s: 0 for s in states}

# Perform First-Visit Monte Carlo Policy Evaluation
returns = {s: [] for s in states}

for episode in episodes:
    # Calculate returns and update the value function
    G = 0
    for t, (state, reward) in reversed(list(enumerate(episode))):
        G = G + reward
        if state not in [x[0] for x in episode[:t]]:
            returns[state].append(G)
            V[state] = np.mean(returns[state])

# Print the value function
print(V)


# ##**Every Visit**

# In[2]:


import numpy as np

# Define the episode
episodes = [
    [('A', 3), ('A', 2), ('B', -4), ('A', 4), ('B', -3)],
    [('B', -2), ('A', 3), ('B', -3)]
    ]

# Define the states
states = ['A', 'B']

# Initialize the value function
V = {s: 0 for s in states}

# Perform Every-Visit Monte Carlo Policy Evaluation
returns = {s: [] for s in states}

for episode in episodes:
    # Calculate returns and update the value function
    G = 0
    for t, (state, reward) in reversed(list(enumerate(episode))):
        G = G + reward
        # Removed the check for first visit
        returns[state].append(G)
        V[state] = np.mean(returns[state])

# Print the value function
print(V)


# In[ ]:




