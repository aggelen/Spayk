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
neurons = [LIFGroup(no_neurons=2, group_label='N', params=cfg.neuron_params),
           LIFGroup(no_neurons=2, group_label='E', params=cfg.neuron_params),
           LIFGroup(no_neurons=2, group_label='I', params=cfg.neuron_params)]
# neurons[0].firing_rate_curve()

#%% Stimulus Synapses
stim_to_neuron = SynapseGroup('external_stim', 'N', cfg.synapse_params, save_channels=True)
stim_to_neuron.AMPA_EXT(gs=1.2e-4, ws=np.random.uniform(low=0.25, high=0.85, size=(2,25)))

syn_N2E = SynapseGroup('N', 'E', cfg.synapse_params, save_channels=True)
syn_N2E.AMPA(gs=4.2e-5, ws=np.random.uniform(low=0.5, high=0.85, size=(2,2)))
syn_N2E.NMDA(gs=4.2e-5, ws=np.random.uniform(low=0.5, high=0.85, size=(2,2)))

syn_N2I = SynapseGroup('N', 'I', cfg.synapse_params, save_channels=True)
syn_N2I.AMPA(gs=2e-5, ws=np.random.uniform(low=0.5, high=0.85, size=(2,2)))
syn_N2I.NMDA(gs=2e-5, ws=np.random.uniform(low=0.5, high=0.85, size=(2,2)))

syn_E2I = SynapseGroup('N', 'I', cfg.synapse_params, save_channels=True)
syn_E2I.AMPA(gs=2e-5, ws=np.random.uniform(low=0.5, high=0.85, size=(2,2)))
syn_E2I.NMDA(gs=2e-5, ws=np.random.uniform(low=0.5, high=0.85, size=(2,2)))

syn_I2E = SynapseGroup('I', 'E', cfg.synapse_params, save_channels=True)
syn_I2E.GABA(gs=4e-5, ws=np.random.uniform(low=0.5, high=0.85, size=(2,2)))

synapses = [stim_to_neuron, syn_N2E, syn_N2I, syn_E2I, syn_I2E]
# synapses[0].synapse_curve()
# synapses = [s_E2E, s_E2I, s_I2E, s_I2I, s_Noise2E, s_Noise2I, s_StimA2A, s_StimB2B]

#%% Build Neural Circuit
params = {'dt': cfg.dt,
          'sim_duration': 1}
nc = NeuralCircuit(neurons, synapses, stimulus, params)

#%% Simulation
nc.keep_alive()

#%% Output Spikes
op_spikes = np.array(nc.problem.output_spikes).T
op_train = SpikeTrain(op_spikes, [0, 1, cfg.dt])

print("\nFiring Rate Op: {}\n".format(np.sum(op_spikes, 1)))
if np.sum(op_spikes):
    op_train.raster_plot()

#%%
nc.problem.stimuli['external_stim'].raster_plot(title="external_stim")


#%%
print(nc.problem.channel_history.keys())

s_AMPA_EXT__CONN_0 = np.array(nc.problem.channel_history['s_AMPA_EXT__CONN_0'])

s_AMPA__CONN_1 = np.array(nc.problem.channel_history['s_AMPA__CONN_1'])
s_NMDA__CONN_2 = np.array(nc.problem.channel_history['s_NMDA__CONN_2'])
s_GABA__CONN_7 = np.array(nc.problem.channel_history['s_GABA__CONN_7'])

plt.figure()
plt.plot(s_AMPA_EXT__CONN_0[:,0,7])
plt.plot(s_AMPA_EXT__CONN_0[:,0,10])
plt.title("s AMPA EXT")

plt.figure()
plt.plot(s_AMPA__CONN_1[:,0,0])
plt.plot(s_AMPA__CONN_1[:,0,1])
plt.title("s AMPA")

plt.figure()
plt.plot(s_NMDA__CONN_2[:,0,0])
plt.plot(s_NMDA__CONN_2[:,0,1])
plt.title("s NMDA")

plt.figure()
plt.plot(s_GABA__CONN_7[:,0,0])
plt.plot(s_GABA__CONN_7[:,0,1])
plt.title("s GABA")

plt.figure()
plt.plot(np.array(nc.problem.I_syn_hist))
plt.title("Synaptic Current")