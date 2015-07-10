import numpy as np
import matplotlib.pylab as plt
from scipy import integrate
import scipy

import scipy.stats



N=100
Tmax=100.0
Nsteps=1000
p=0.1

Phi = lambda z: 0.5 + 0.5 * scipy.special.erf(z / np.sqrt(2)) #area of gaussian till z

a=np.ones([N,N])
#print a

for i in xrange(N):
    if np.random.rand()>Phi(-1):
        a[i,:]=scipy.stats.truncnorm.rvs(-1, np.inf, size=N)+1
    else:
        a[i,:]=scipy.stats.truncnorm.rvs(-1*np.inf, -1, size=N)+1
        
#print a
a=a*np.random.binomial(1,p,[N,N])/p 


fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(a,interpolation='none',vmin=-4, vmax=4)
ax.set_aspect('equal')

#cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
#cax.get_xaxis().set_visible(False)
#cax.get_yaxis().set_visible(False)
#cax.patch.set_alpha(0.5)
#cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()

print np.mean(a),np.var(a)