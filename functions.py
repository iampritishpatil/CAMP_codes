import numpy as np
import matplotlib.pylab as plt
from scipy import integrate
import scipy
import scipy.stats
import scipy.ndimage.filters




Phi = lambda z: 0.5 + 0.5 * scipy.special.erf(z / np.sqrt(2)) #area of gaussian till z

def create_matrix(p=0.1,N=100,mean=1):
    a=np.zeros([N,N])
    for i in xrange(N):
        if np.random.rand()>Phi(-1*mean):
            a[i,:]=scipy.stats.truncnorm.rvs(-1*mean, np.inf, size=N)+1*mean
        else:
            a[i,:]=scipy.stats.truncnorm.rvs(-1*np.inf, -1*mean, size=N)+1*mean
    a=a*np.random.binomial(1,p,[N,N])/(p*N)
    return a

def plot_matrix(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    vmax=np.max(np.max(matrix),-1*np.min(matrix))
    plt.imshow(matrix,interpolation='none',cmap=plt.get_cmap('seismic'),vmin=-1*vmax,vmax=vmax)
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


def spike_to_rate(spiketimes,tmax,filter_size):
    arr=np.zeros([np.size(spiketimes),tmax*1000])
    for i,data in enumerate(spiketimes):
        #print i
        temp_mat=np.zeros(tmax*1000)
        temp_mat[(data*1000).astype(int)]=1000
        scipy.ndimage.filters.gaussian_filter1d(temp_mat,filter_size,mode='constant',output=arr[i])
        #plt.plot(arr[i])
        #plt.show()
    return arr


