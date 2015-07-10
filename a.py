import nest
network = nest.Create('iaf_neuron',1000)
conn_dict = {'rule': 'pairwise_bernoulli', 'p': 1.0}
syn_dict ={"model": "static_synapse", "weight":{'distribution': 'normal', 'mu': 5.0, 'sigma': 1.0}, "delay":1.0}
nest.Connect(network,network,conn_dict,syn_spec=syn_dict)

pg = nest.Create('poisson_generator',1,{'rate':5000.0})
par = nest.Create('parrot_neuron',1)
nest.DivergentConnect(pg,network)
nest.Connect(pg,par)
SD = nest.Create("spike_detector")
SD1 = nest.Create("spike_detector")
nest.ConvergentConnect(network,SD)
nest.Connect(par,SD1)
nest.Simulate(1000)
spikes=nest.GetStatus(SD)[0]["events"]["times"]
ids=nest.GetStatus(SD)[0]["events"]["senders"]

spikes1=nest.GetStatus(SD1)[0]["events"]["times"]
ids1=nest.GetStatus(SD1)[0]["events"]["senders"]

