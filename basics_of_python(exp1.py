# -*- coding: utf-8 -*-
"""RL_Exp_1_Basics_Of_Python.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Khushal21csu188/RLLAB-SEM-5/blob/main/RL_Exp_1_Basics_Of_Python.ipynb
"""

!python --version

x=3
print(type(x))
print(x)

t=True
f=False
print(type(t))
print(t and f)
print(t or f)

x=5
y=10
z=x+y
print("Addition of %s and %s is %s"%(x,y,z))

x=5
y=10
z=x+y
str=("Addition of {} and {} ={}".format(x,y,z))
print(str)

fruits = ['apple', 'banana', 'cherry']
fruits.append("orange")
print(fruits)

nums=list(range(10))
nums[2:4]=[8,9]
print(nums)

animals=['cat','dog','monkey']
for animal in animals:
  print(animal)

animals=['cat','dog','monkey']
for idx,animal in enumerate (animals):
  print('#{}: {}'.format(idx + 1,animal))

nums=list(range(11))
squares= []
for x in nums:
    squares.append(x**2)
print(squares)

nums=list(range(11))
even_squares=[x**2 for x in nums if x%2==0]
print(even_squares)