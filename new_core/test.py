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

from experiment_config_wang import Wang2002Config as cfg

from spayk.Stimuli import PoissonActivity, SpikeTrain

#%% Stimulus
stimulus = [PoissonActivity(cfg.no_noise_E, cfg.freq_noise_E, cfg.stim_time_params,'noiseE'),
            PoissonActivity(cfg.no_noise_I, cfg.freq_noise_E, cfg.stim_time_params, 'noiseI'),
            PoissonActivity(cfg.no_stim_A, cfg.freq_stim_A, cfg.stim_time_params, 'stimA'),
            PoissonActivity(cfg.no_stim_B, cfg.freq_stim_B, cfg.stim_time_params,'stimB')]

#%% Neurons
neurons = [LIFGroup(no_neurons=cfg.no_exc, group_label='E', params=cfg.exc_neuron_params),
           LIFGroup(no_neurons=cfg.no_inh, group_label='I', params=cfg.inh_neuron_params)]

# Exc to exc:  cols from, rows to
#    |   A       B       N
#  ---------------------------------
#  A |   w+      w-      w-
#  B |   w-      w+      w-
#  N |   1       1       1

WA = np.hstack([np.full((cfg.no_A, cfg.no_A), cfg.w_plus),
                np.full((cfg.no_A, cfg.no_B), cfg.w_minus),
                np.full((cfg.no_A, cfg.no_N), cfg.w_minus)])
WB = np.hstack([np.full((cfg.no_B, cfg.no_A), cfg.w_minus),
                np.full((cfg.no_B, cfg.no_B), cfg.w_plus),
                np.full((cfg.no_B, cfg.no_N), cfg.w_minus)])
WN = np.hstack([np.full((cfg.no_N, cfg.no_A), 1.0),
                np.full((cfg.no_N, cfg.no_B), 1.0),
                np.full((cfg.no_N, cfg.no_N), 1.0)])
W_exc2exc = np.vstack([WA,WB,WN])
np.fill_diagonal(W_exc2exc, 0.0)   # Each neuron receives inputs from all other neurons, but with structured synaptic weights.

#%% Recurrent Synapses
#                  0    1        2           3         4       5
# state_vector = ['V', 'sAMPA', 'sAMPA_EXT', 'xNMDA', 'sNMDA', 'sGABA']
# E2E Synapses
s_E2E = SynapseGroup('E', 'E', cfg.synapse_params)
s_E2E.AMPA(gs=cfg.g_ampa_ext2exc, ws=W_exc2exc, state_label='sAMPA')
s_E2E.NMDA(gs=cfg.g_nmda_ext2exc, ws=W_exc2exc, state_label='xNMDA, sNMDA')

# E2I Synapses
s_E2I = SynapseGroup('E', 'I', cfg.synapse_params)
s_E2I.AMPA(gs=cfg.g_ampa_ext2inh, ws=np.ones((cfg.no_inh, cfg.no_exc)), state_label='sAMPA')
s_E2I.NMDA(gs=cfg.g_nmda_ext2inh, ws=np.ones((cfg.no_inh, cfg.no_exc)), state_label='xNMDA, sNMDA')

# I2E Synapses
s_I2E = SynapseGroup('I', 'E', cfg.synapse_params)
s_I2E.GABA(gs=cfg.g_gaba_inh2exc, ws=np.ones((cfg.no_exc, cfg.no_inh)), state_label='sGABA')

wI2I = np.ones((cfg.no_inh, cfg.no_inh))
np.fill_diagonal(wI2I, 0.0)
# I2I Synapses
s_I2I = SynapseGroup('I', 'I', cfg.synapse_params)
s_I2I.GABA(gs=cfg.g_gaba_inh2inh, ws=wI2I, state_label='sGABA')

#%% Stimulus Synapses
s_Noise2E = SynapseGroup('noiseE', 'E', cfg.synapse_params)
s_Noise2E.AMPA_EXT(gs=cfg.g_ampa_ext2exc, ws=np.ones((cfg.no_exc, cfg.no_noise_E)), state_label='sAMPA_EXT')

s_Noise2I = SynapseGroup('noiseI', 'I', cfg.synapse_params)
s_Noise2I.AMPA_EXT(gs=cfg.g_ampa_ext2inh, ws=np.ones((cfg.no_inh, cfg.no_noise_I)), state_label='sAMPA_EXT')

s_StimA2A = SynapseGroup('stimA', 'E[0:240]', cfg.synapse_params)
s_StimA2A.AMPA_EXT(gs=cfg.g_ampa_ext2exc, ws=np.ones((cfg.no_A, cfg.no_stim_A)), state_label='sAMPA_EXT')

s_StimB2B = SynapseGroup('stimB', 'E[240:480]', cfg.synapse_params)
s_StimB2B.AMPA_EXT(gs=cfg.g_ampa_ext2exc, ws=np.ones((cfg.no_B, cfg.no_stim_B)), state_label='sAMPA_EXT')

synapses = [s_Noise2E, s_Noise2I, s_StimA2A, s_StimB2B]
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
op_train.raster_plot()

#%%
# nc.problem.stimuli['noiseE'].raster_plot(title="NoiseE")
# nc.problem.stimuli['noiseI'].raster_plot(title="NoiseI")
# # nc.problem.stimuli['stimA'].raster_plot(title="stimA")
# nc.problem.stimuli['stimB'].raster_plot(title="stimB")

# V = np.array(nc.problem.V_hist).T
# plt.figure()
# plt.plot(V[5,:])
