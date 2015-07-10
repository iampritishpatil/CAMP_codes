from functions import *
import brian
import numpy as np
import pylab as plt

N=100
p=0.1

taum = 20 * brian.ms          # membrane time constant
taue = 5 * brian.ms          # excitatory synaptic time constant
taui = 10 * brian.ms          # inhibitory synaptic time constant
Vt = -50 * brian.mV          # spike threshold
Vr = -60 * brian.mV          # reset value
El = -49 * brian.mV          # resting potential
we = 1 * brian.mV # excitatory synaptic weight
wi = 1 * brian.mV # inhibitory synaptic weight


mat=create_matrix(p,N)

map_exc=lambda i,j:we*(mat[i,j]) if mat[i,j]>0 else 0*brian.mV
map_inh=lambda i,j:wi*(mat[i,j]) if mat[i,j]<0 else 0*brian.mV

eqs = brian.Equations('''

        dV/dt  = (ge-gi-(V-El))/taum : volt
        dge/dt = -ge/taue            : volt
        dgi/dt = -gi/taui            : volt
        ''')
neurons = brian.NeuronGroup(N, model=eqs, threshold=Vt, reset=Vr)

exc=brian.Connection(neurons, neurons, 'ge', sparseness=1, weight=map_exc, threshold=Vt, reset=Vr,refractory=2*brian.ms)

inh=brian.Connection(neurons, neurons, 'gi', sparseness=1, weight=map_exc, threshold=Vt, reset=Vr,refractory=2*brian.ms)

spikes=brian.SpikeMonitor(neurons)

neurons.V = Vr + np.random.rand(N) * (Vt - Vr)*1.25

brian.run(1*brian.second)

brian.raster_plot(spikes)
plt.show()