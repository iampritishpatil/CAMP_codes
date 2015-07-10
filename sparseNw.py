import nest
import numpy as np
import pylab as pl

conn = 0.1
N = 100
f = 0.5 # Percentage of Exc neurons , 1 -f = % of inhibitory neurons
simTime = 15000
nest.ResetKernel()
wtE = np.zeros((int(N*f),int(conn*N))) # Rows = all outgoing connections
wtI = np.zeros((int(N*(1-f)),int(conn*N)))

#initEx = np.random.uniform(0,1,N)

nwE = nest.Create('iaf_neuron',int(N*f),params={'I_e':1000*np.random.rand(),"V_m": -70+10*np.random.rand()})
nwI = nest.Create('iaf_neuron',int(N*(1-f)),params={'I_e':1000*np.random.rand()})
# Mean = 1/sqrt(N), var = 1/N
wtE[:] = np.random.normal(0,1./(N),(int(N*f),int(N*conn))) + 1.0/N
wtI[:] = np.random.normal(0,1./(N),(int(N*(1-f)),int(N*conn))) + 1.0/N

for i,exc in enumerate(nwE):
	inds = np.random.randint(1,N,size=int(N*conn))
	nest.DivergentConnect([exc],inds.tolist(),weight=wtE[i].tolist(),delay=np.ones((np.shape(wtE[i]))).tolist())
for j,inh in enumerate(nwI):
	inds = np.random.randint(1,N,size=int(N*conn))
	nest.DivergentConnect([inh],inds.tolist(),weight=wtI[i].tolist(),delay=np.ones((np.shape(wtI[i]))).tolist())


WtMat = np.zeros((N,N))

# Generate the weight matrix

for i,exc in enumerate(nwE):
	conn = nest.GetConnections([exc],nwE+nwI,synapse_model='static_synapse')
	wt = nest.GetStatus(conn,keys='weight')
	ids = np.array(conn)[:,1] # Destinations from exc
	WtMat[i,ids] = wt 
for j,inh in enumerate(nwI):
	conn = nest.GetConnections([inh],nwE+nwI,synapse_model='static_synapse')
	wt = nest.GetStatus(conn,keys='weight')
	ids = np.array(conn)[:,1] # Destinations from exc
	WtMat[j+len(nwE),ids] = wt 

# Give a transient input
currgen = nest.Create('dc_generator',1,{'start':0.,'stop':float(simTime),'amplitude':300. })
#pg = nest.Create('poisson_generator',1,{'rate':)
nest.DivergentConnect(currgen,nwE)
nest.DivergentConnect(currgen,nwI)

sdE = nest.Create('spike_detector',1)
sdI = nest.Create('spike_detector',1)


nest.ConvergentConnect(nwE,sdE)
nest.ConvergentConnect(nwI,sdI)
# Eigen values of the connectivity matrix

nest.Simulate(simTime)

timesE = nest.GetStatus(sdE)[0]['events']['times']
idE = nest.GetStatus(sdE)[0]['events']['senders']

timesI = nest.GetStatus(sdI)[0]['events']['times']
idI = nest.GetStatus(sdI)[0]['events']['senders']


pl.figure()
pl.plot(timesE,idE,'b.',label='Exc')
pl.plot(timesI,idI,'r.',label='Inh')
pl.xlim(0,simTime)
pl.legend()
pl.show()

w,e = np.linalg.eig(WtMat)
pl.figure()
pl.plot(w.real,w.imag,'b.')
 




 



