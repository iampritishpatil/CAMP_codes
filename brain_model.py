from functions import *
import brian
import numpy as np
import pylab as plt

N=1000
p=0.03

taum = 20 * brian.ms          # membrane time constant
taue = 5 * brian.ms          # excitatory synaptic time constant
taui = 10 * brian.ms          # inhibitory synaptic time constant
Vt = -40 * brian.mV          # spike threshold
Vr = -65 * brian.mV          # reset value
El = -65 * brian.mV          # resting potential
we = 40.0/N /p * brian.mV # excitatory synaptic weight
wi = 40.0/N /p * brian.mV # inhibitory synaptic weight


mat=create_matrix(p,N)

map_exc=lambda i,j:we*(mat[i,j]) if mat[i,j]>0 else 0*brian.mV
map_inh=lambda i,j:wi*(mat[i,j]) if mat[i,j]<0 else 0*brian.mV

eqs = brian.Equations('''

        dV/dt  = (ge+gi-(V-El)+I)/taum  : volt
        dge/dt = -ge/taue            : volt
        dgi/dt = -gi/taui            : volt
        I : volt
        ''')
neurons = brian.NeuronGroup(N, model=eqs, threshold=Vt, reset=Vr)

neurons.I=np.ones(N)*40*brian.mV

exc=brian.Connection(neurons, neurons, 'ge', sparseness=1, weight=map_exc, threshold=Vt, reset=Vr,refractory=2*brian.ms,delay=1*brian.ms)

inh=brian.Connection(neurons, neurons, 'gi', sparseness=1, weight=map_inh, threshold=Vt, reset=Vr,refractory=2*brian.ms,delay=1*brian.ms)


spikes=brian.SpikeMonitor(neurons)

neurons.V = Vr + np.random.rand(N) * (Vt - Vr)*0.9

print neurons.I

brian.run(0.11*brian.second)
#plt.subplot(211)
neurons.I=0*np.ones(N)*brian.mV

brian.run(0.5*brian.second)

brian.raster_plot(spikes)
plt.show()

a=spikes.getspiketimes()
b=a.values()

