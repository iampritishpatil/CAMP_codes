# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:40:22 2015

@author: pritish
"""

import numpy as np
import matplotlib.pylab as pl
from scipy import integrate
import scipy







#size
N=100
Tmax=100.0
Nsteps=10000
p=0.1

Phi = lambda z: 0.5 + 0.5 * scipy.special.erf(z / sqrt(2)) #area of gaussian till z

connectivity=((np.random.randn(N,N)+1)/N)*np.random.binomial(1,p,[N,N])/p 
#connectivity=((np.random.randn(N,N)+1)/N)
#connectivity=a


#sparser=np.random.binomial(1,0.1,[N,N])





w,v=np.linalg.eig(connectivity)

vinv = np.linalg.inv(np.matrix(v))
wa=np.dot(np.dot(vinv,connectivity),v)



pl.scatter(w.real,w.imag)
pl.show()
def f(y,t):
    return -1*y+np.dot(y,connectivity) +0.1

t = np.linspace(0, Tmax,  Nsteps)
r0 = np.random.rand(N) *100

X, infodict = integrate.odeint(f, r0, t, full_output=True)
print infodict['message']  
pl.plot(t,X)
pl.show()
Xnew=np.dot(X,v)
pl.plot(t,Xnew)
pl.show()

