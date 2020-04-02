# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:08:52 2020

@author: gabin
"""

import numpy as np
import matplotlib.pyplot as plt
import math


h=0.5
def fonction(x):
    f=-x
    return (f)


def euler(u,h,f):
    U=u+h*f
    return (U)

u=np.zeros([10])
u[0]=1
for i in range (0,9):
    f=fonction(u[i])
    u[i+1]=euler(u[i],h,f)

tvec=np.linspace(0,10,100)
T=np.zeros([100])
U=np.zeros([100])
for i in range (0,100):
    T[i]=math.exp(-tvec[i])
    U[i]=(1-h)**(tvec[i])
plt.plot(u,color='green')
plt.plot(tvec,T,color='red')
plt.plot(tvec,U,color='black')
plt.show()

