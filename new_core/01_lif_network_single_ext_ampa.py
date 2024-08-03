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
neurons = [LIFGroup(no_neurons=1, group_label='N', params=cfg.exc_neuron_params)]
# neurons[0].firing_rate_curve()

#%% Stimulus Synapses
stim_to_neuron = SynapseGroup('external_stim', 'N', cfg.synapse_params, save_channels=True)
stim_to_neuron.AMPA_EXT(gs=5.8*1e-5, ws=np.ones((1, 25)))

synapses = [stim_to_neuron]
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
# nc.problem.channel_history.keys()

s_AMPA_EXT__CONN_0 = np.array(nc.problem.channel_history['s_AMPA_EXT__CONN_0'])

plt.figure()
plt.plot(s_AMPA_EXT__CONN_0[:,0,7])
plt.plot(s_AMPA_EXT__CONN_0[:,0,10])

plt.figure()
plt.plot(np.array(nc.problem.I_syn_hist))
