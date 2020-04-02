# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:40:52 2020

@author: HAFOLAHBI
"""

# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
​
Ceci est un script temporaire.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
n = int(input("Enter the number of points, n = ")) # I just made little input
def pts_interp(n):
    x=np.zeros([n])
    y=np.zeros([n])
    for i in range (0,n):
        x[i]=math.pi/2*i
        y[i]=math.sin(x[i])
    return(x,y)
x,y=pts_interp(n)
print (x)
print (y)
def notre_lagrange(t,x,y):
    p=0
    n=len(x)
    for i in range (0,n):
        L=1
        for j in range (0,n):
            if j!=i:
                L=(t-x[j])/(x[i]-x[j])*L
        p=p+y[i]*L
        return p
tvec=np.linspace(0,(n-1)*math.pi/2,100)
pvec=np.zeros([100])
T=np.zeros([100])
d=np.zeros([100])
for i in range (0,100):
    pvec[i]=notre_lagrange(tvec[i],x,y)
    T[i]=math.sin(tvec[i])
    d[i]=abs(pvec[i]-T[i])
plt.plot(tvec,pvec,color='brown') #I just change the color
plt.scatter(x,y,color='red')
plt.plot(tvec,T,color='green')
plt.show()
D=max(d)
print(D)