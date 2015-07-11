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

neurons.I=np.ones(N)*30*brian.mV

exc=brian.Connection(neurons, neurons, 'ge', weight=map_exc, threshold=Vt, reset=Vr,delay=1*brian.ms)

inh=brian.Connection(neurons, neurons, 'gi', weight=map_inh, threshold=Vt, reset=Vr,delay=1*brian.ms)


spikes=brian.SpikeMonitor(neurons)

neurons.V = Vr + np.random.rand(N) * (Vt - Vr)*1.1

#print neurons.I

brian.run(0.11*brian.second)
#plt.subplot(211)
neurons.I=0*np.ones(N)*brian.mV

brian.run(1*brian.second)

brian.raster_plot(spikes,marker='.',color='k')
plt.show()

a=spikes.getspiketimes()
b=a.values()

