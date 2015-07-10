import numpy as np
import matplotlib.pylab as plt
from scipy import integrate
import scipy
import scipy.stats





Phi = lambda z: 0.5 + 0.5 * scipy.special.erf(z / np.sqrt(2)) #area of gaussian till z

def create_martix(p=0.1,N=100):
    a=np.zeros([N,N])
    for i in xrange(N):
        if np.random.rand()>Phi(-1):
            a[i,:]=scipy.stats.truncnorm.rvs(-1, np.inf, size=N)+1
        else:
            a[i,:]=scipy.stats.truncnorm.rvs(-1*np.inf, -1, size=N)+1
    a=a*np.random.binomial(1,p,[N,N])/(p*N)
    return a

def plot_matrix(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    vmax=np.max(np.max(matrix),-1*np.min(matrix))
    plt.imshow(matrix,cmap=plt.get_cmap('seismic'),vmin=-1*vmax,vmax=vmax)#interpolation='none',
    ax.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    plt.show()



def plot_eigenvalues(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Eigenvalues')
    ax.set_aspect('equal')
    w,v=np.linalg.eig(matrix)
    plt.scatter(w.real,w.imag)
    plt.show()





