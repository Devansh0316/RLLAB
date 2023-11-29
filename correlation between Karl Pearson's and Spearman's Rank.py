#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Khushal21csu188/RLLAB-SEM-5/blob/main/RL_Exp_2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt
X = np.array([1,2,3,4])
Y = np.array([4,5,6,7])
r = np.corrcoef(X,Y)
plt.scatter(X,Y)
plt.show()
print(r)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
X = np.array([4,5,6,7])
Y = np.array([7,6,5,4])
r = np.corrcoef(X,Y)
plt.scatter(X,Y)
plt.show()
print(r)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
X = np.array([14,2,-1,5,4,11,-10])
Y = np.array([8,-4,1,3,-1,3,-1])
r = np.corrcoef(X,Y)
plt.scatter(X,Y)
plt.show()
print(r)

