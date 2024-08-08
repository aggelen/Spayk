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

#%% Stim on the paper
resample_period = 50   # every 50 ms
c = 12.8
mu_0 = 40
sigma = 4

def mean_stimulus(mu_0, coherence):
    ro_a = mu_0 / 100
    ro_b = mu_0 / 100
    mu_a = mu_0 + ro_a*coherence
    mu_b = mu_0 - ro_b*coherence
    return mu_a, mu_b

def sample_stimulus(mu_a, mu_b, sigma, resample_period=50e-3, dt=cfg.dt):
    #every 50ms >> resample
    sa = np.random.normal(mu_a, sigma)
    sb = np.random.normal(mu_b, sigma)
    
    return np.full(int(resample_period/dt), sa), np.full(int(resample_period/dt), sb)

mu_a, mu_b = mean_stimulus(mu_0, c)

stim_a = []
stim_b = []

for i in range(0,int(int(cfg.sim_duration*1e3)/resample_period)):
    a, b = sample_stimulus(mu_a, mu_b, sigma)
    stim_a.append(a)
    stim_b.append(b)

stim_a, stim_b = np.hstack(stim_a), np.hstack(stim_b)

stim_a[:int(len(stim_a)/2)] = 0.0
stim_b[:int(len(stim_a)/2)] = 0.0

plt.figure()
plt.plot(stim_a)
plt.plot(stim_b)
plt.xlabel('Time (ms)')
plt.ylabel('Sample Stim')
plt.legend(['pop_a', 'pop_b'])

#%% Stimulus
stimulus = [PoissonActivity(cfg.no_noise_E, cfg.freq_noise_E, cfg.stim_time_params,'noiseE'),
            PoissonActivity(cfg.no_noise_I, cfg.freq_noise_I, cfg.stim_time_params, 'noiseI'),
            PoissonActivity(cfg.no_stim_A, stim_a, cfg.stim_time_params, 'stimA'),
            PoissonActivity(cfg.no_stim_B, stim_b, cfg.stim_time_params,'stimB')]

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
s_E2E.AMPA(gs=cfg.g_ampa_exc2exc, ws=W_exc2exc)
s_E2E.NMDA(gs=cfg.g_nmda_exc2exc, ws=W_exc2exc)

# E2I Synapses
s_E2I = SynapseGroup('E', 'I', cfg.synapse_params)
s_E2I.AMPA(gs=cfg.g_ampa_exc2inh, ws=np.ones((cfg.no_inh, cfg.no_exc)))
s_E2I.NMDA(gs=cfg.g_nmda_exc2inh, ws=np.ones((cfg.no_inh, cfg.no_exc)))

# I2E Synapses
s_I2E = SynapseGroup('I', 'E', cfg.synapse_params)
s_I2E.GABA(gs=cfg.g_gaba_inh2exc, ws=np.ones((cfg.no_exc, cfg.no_inh)))

wI2I = np.ones((cfg.no_inh, cfg.no_inh))
np.fill_diagonal(wI2I, 0.0)

# I2I Synapses
s_I2I = SynapseGroup('I', 'I', cfg.synapse_params)
s_I2I.GABA(gs=cfg.g_gaba_inh2inh, ws=wI2I)

#%% Stimulus Synapses
s_Noise2E = SynapseGroup('noiseE', 'E', cfg.synapse_params)
s_Noise2E.AMPA_EXT(gs=cfg.g_ampa_ext2exc, ws=np.ones((cfg.no_exc, cfg.no_noise_E)))

s_Noise2I = SynapseGroup('noiseI', 'I', cfg.synapse_params)
s_Noise2I.AMPA_EXT(gs=cfg.g_ampa_ext2inh, ws=np.ones((cfg.no_inh, cfg.no_noise_I)))

s_StimA2A = SynapseGroup('stimA', 'E[0:240]', cfg.synapse_params)
s_StimA2A.AMPA_EXT(gs=cfg.g_ampa_ext2exc, ws=np.ones((cfg.no_A, cfg.no_stim_A)))

s_StimB2B = SynapseGroup('stimB', 'E[240:480]', cfg.synapse_params)
s_StimB2B.AMPA_EXT(gs=cfg.g_ampa_ext2exc, ws=np.ones((cfg.no_B, cfg.no_stim_B)))

# synapses = [s_Noise2E, s_Noise2I]
synapses = [s_Noise2E, s_Noise2I, s_StimA2A, s_StimB2B]
# synapses = [s_Noise2E, s_Noise2I, s_StimA2A, s_StimB2B, s_E2E]
# synapses = [s_E2E, s_E2I, s_I2E, s_I2I, s_Noise2E, s_Noise2I, s_StimA2A, s_StimB2B]
#%% Build Neural Circuit
params = {'dt': cfg.dt,
          'sim_duration': cfg.sim_duration}
wang_nc = NeuralCircuit(neurons, synapses, stimulus, params)

#%% Simulation
wang_nc.keep_alive()

#%% Output Spikes
op_spikes = np.array(wang_nc.problem.output_spikes).T
spike_loc_A = np.argwhere(op_spikes[:240])
spike_loc_B = np.argwhere(op_spikes[240:480])
op_train = SpikeTrain(op_spikes, [0, 1, cfg.dt])

print("\nFiring Rate Op: {}\n".format(np.sum(op_spikes, 1)))
if np.sum(op_spikes):
    op_train.raster_plot()
    
#%%
fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained', gridspec_kw={'height_ratios': [1, 1]})
axs[0].plot(spike_loc_A[:,1], spike_loc_A[:,0], '.', markersize=2, color='darkred')
axs[0].set(ylabel='population A', ylim=(0, 240))

axs[1].plot(spike_loc_B[:,1], spike_loc_B[:,0], '.', markersize=2, color='darkblue')
axs[1].set(ylabel='population B', ylim=(0, 240))

# plt.figure()
# plt.plot(SME2I.t / ms, SME2I.i, '.', markersize=2, color='darkred')

#%%
# wang_nc.problem.stimuli['noiseE'].raster_plot(title="noiseE")

#%%
# I_syn = np.array(wang_nc.problem.I_syn_hist)
# plt.figure()
# plt.plot(I_syn[4])