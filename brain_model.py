from functions import *
import brian
import numpy as np
import pylab as plt
from brian.library.random_processes import *

N=100
p=0.1

taum = 20 * brian.ms          # membrane time constant
taue = 5 * brian.ms          # excitatory synaptic time constant
taui = 10 * brian.ms          # inhibitory synaptic time constant
Vt = -50 * brian.mV          # spike threshold
Vr = -65 * brian.mV          # reset value
El = -65 * brian.mV          # resting potential
we = 100.0 * brian.mV # excitatory synaptic weight ##double to compensate for the smaller tau so that on average similar amount of current are injected
wi = 50.0 * brian.mV # inhibitory synaptic weight


mat=create_matrix(p=p,N=N,mean=1)

map_exc=lambda i,j:we*(mat[i,j]) if mat[i,j]>0 else 0*brian.mV
map_inh=lambda i,j:wi*(mat[i,j]) if mat[i,j]<0 else 0*brian.mV

eqs = brian.Equations('''

        dV/dt  = (ge+gi-(V-El)+I+J)/taum  : volt
        dge/dt = -ge/taue            : volt
        dgi/dt = -gi/taui            : volt
        I : volt
        ''')
        
eqs+=OrnsteinUhlenbeck('J',mu=0*brian.mV,sigma=10*brian.mV,tau=10*brian.ms)

neurons = brian.NeuronGroup(N, model=eqs, threshold=Vt, reset=Vr,refractory=2*brian.ms)



exc=brian.Connection(neurons, neurons, 'ge', weight=map_exc, threshold=Vt, reset=Vr,delay=1*brian.ms)

inh=brian.Connection(neurons, neurons, 'gi', weight=map_inh, threshold=Vt, reset=Vr,delay=1*brian.ms)


spikes=brian.SpikeMonitor(neurons)

neurons.V = Vr + np.random.rand(N) * (Vt - Vr)*1.1

tper=2
numtimes=1
tmax =tper*numtimes
#print neurons.I
for ti in range(numtimes):
    neurons.I=np.random.rand(N)*40*brian.mV
    brian.run(tper*0.5*brian.second)
    neurons.I=np.random.rand(N)*15*brian.mV
    brian.run(tper*0.5*brian.second)
    print ti
    
#plt.subplot(211)
#neurons.I=0*np.ones(N)*brian.mV

#brian.run(5*brian.second)
#plt.subplot(411)
#brian.raster_plot(spikes,marker='.',color='k')
#plt.show()

a=spikes.getspiketimes()
b=a.values()
plt.subplot(411)
rate_mat=spike_to_rate(spiketimes=b,tmax=tmax,filter_size=200)
for i,r in enumerate(rate_mat):
    plt.plot(r)
#plt.show()

plt.title('Firing rates of different neurons')
plt.ylabel('Frequency (Hz)')
plt.subplot(412)
w,v=np.linalg.eig(mat)
vinv = np.linalg.inv(np.matrix(v))
Xnew=np.dot(rate_mat.T,v)


for i,r in enumerate(Xnew.T[np.where(w!=w.max())]):
    plt.plot(r)

plt.title('Components resolved from weight matrix')
plt.ylabel('Relative units')

plt.subplot(413)
for i,r in enumerate(Xnew.T[np.where(w==w.max())]):
    if r.sum()<0:
        plt.plot(-1*r,label='From weight matrix')
    else:
        plt.plot(r,label='From weight matrix')
    
    
plt.ylabel('Relative units')
X=np.dot(rate_mat,rate_mat.T)
o,k=np.linalg.eig(X)


XXnew = np.dot(rate_mat.T,k)
for i,r in enumerate(XXnew.T[np.where(o==o.max())]):
    if r.sum()<0:
        plt.plot(-1*r,label='From Component analysis')
    else:
        plt.plot(r,label='From Component analysis')

plt.title('Comprision of the components resolved')
plt.ylabel('Relative units')


plt.subplot(414)

for i,r in enumerate(XXnew.T[np.where(o!=o.max())]):
    plt.plot(r)
plt.title('Other components resolved from analysis')
plt.ylabel('Relative units')

plt.xlabel('Time (ms)')

plt.tight_layout()

plt.show()






