#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:12:33 2024

@author: gelenag

Scenario:
    16 Pyramidal cells, excitatory
    4 Interneurons, inhibitory
    10 Neurons, Poisson, Random Noise
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from spayk.Architechtures import NeuralCircuit
from spayk.Neurons import LIFGroup
from spayk.Synapses import SynapseGroup

from experiment_config import Wang2002Config as cfg

from spayk.Stimuli import PoissonActivity, SpikeTrain

#%% Stimulus
stimulus = [PoissonActivity(25, 25, cfg.stim_time_params,'external_stim')]

#%% Neurons
no_neurons = 1
neurons = [LIFGroup(no_neurons=no_neurons, group_label='E', params=cfg.exc_neuron_params),
           LIFGroup(no_neurons=no_neurons, group_label='I', params=cfg.inh_neuron_params)]

# neurons[0].firing_rate_curve()

#%% Stimulus Synapses
stim_to_neuron = SynapseGroup('external_stim', 'E', cfg.synapse_params)
stim_to_neuron.AMPA_EXT(gs=cfg.g_ampa_ext2exc, ws=np.ones((no_neurons, 25)), state_label='sAMPA_EXT')

E_to_I = SynapseGroup('E', 'I', cfg.synapse_params)
E_to_I.AMPA(gs=cfg.g_ampa_exc2inh, ws=np.ones((1, 1)), state_label='sAMPA')

synapses = [stim_to_neuron, E_to_I]
# synapses[0].synapse_curve()
# synapses = [s_E2E, s_E2I, s_I2E, s_I2I, s_Noise2E, s_Noise2I, s_StimA2A, s_StimB2B]

#%% Build Neural Circuit
params = {'dt': cfg.dt}
nc = NeuralCircuit(neurons, synapses, stimulus, params)

#%% Simulation
stop_time = 1
nc.keep_alive(tsim=stop_time)

#%% Output Spikes
op_spikes = np.array(nc.problem.output_spikes).T
op_train = SpikeTrain(op_spikes, [0, 1, cfg.dt])

print("Firing Rate Op: {}".format(np.sum(op_spikes)))
if np.sum(op_spikes):
    op_train.raster_plot()

#%%
nc.problem.stimuli['external_stim'].raster_plot(title="external_stim")


#%%
plt.figure()
plt.plot(np.array(nc.problem.sAmpa_hist)[:,0,7])
plt.plot(np.array(nc.problem.sAmpa_hist)[:,0,10])

plt.figure()
plt.plot(np.array(nc.problem.I_syn_hist))
