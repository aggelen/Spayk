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

#%% Neurons
neurons = [LIFGroup(no_neurons=cfg.no_exc, group_label='E', params=cfg.exc_neuron_params)]

#%% Build Neural Circuit
params = {'dt': cfg.dt,
          'sim_duration': cfg.sim_duration}
wang_nc = NeuralCircuit(neurons, synapses, stimulus, params)

#%% Simulation
wang_nc.keep_alive()

#%%
def population_firing_rates(spikes, dt, time_window, shift):
    window_len = int(time_window / dt)
    shift_len = int(shift / dt)
    rates = []
    stop = int(spikes.shape[1]-window_len)
    shifts = np.arange(0, stop, shift_len)
    for i in shifts:
        windowed_spikes = spikes[:,i:i+window_len]
        rates.append(np.sum(windowed_spikes)/time_window/spikes.shape[0])
    
    return np.array(rates)

#%% Output Spikes
op_spikes = np.array(wang_nc.problem.output_spikes).T
spike_loc_A = np.argwhere(op_spikes[:240])
spike_loc_B = np.argwhere(op_spikes[240:480])
spike_loc_I = np.argwhere(op_spikes[1600:])
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

# axs[1].plot(spike_loc_I[:,1], spike_loc_I[:,0], '.', markersize=2, color='darkgreen')
# axs[1].set(ylabel='population I', ylim=(0, 240))

# plt.figure()
# plt.plot(SME2I.t / ms, SME2I.i, '.', markersize=2, color='darkred')

#%%
# wang_nc.problem.stimuli['noiseE'].raster_plot(title="noiseE")

#%%
# I_syn = np.array(wang_nc.problem.I_syn_hist)
# plt.figure()
# plt.plot(I_syn[4])

#%%
rate_E = population_firing_rates(op_spikes[:1600], cfg.dt, 50e-3, 5e-3)
rate_I = population_firing_rates(op_spikes[1600:], cfg.dt, 50e-3, 5e-3)
plt.figure()
plt.plot(rate_E, color='darkred')
plt.plot(rate_I, color='darkblue')

#%%
rate_A = population_firing_rates(op_spikes[:240], cfg.dt, 100e-3, 5e-3)
rate_B = population_firing_rates(op_spikes[240:480], cfg.dt, 100e-3, 5e-3)
plt.figure()
plt.plot(rate_A, color='darkred')
plt.plot(rate_B, color='darkblue')

#%%
sAMPAext_hist = np.array(wang_nc.problem.log['s_AMPA_ext'])
plt.figure()
plt.plot(sAMPAext_hist, color='darkred')
plt.figure()
IAMPAext_hist = np.array(wang_nc.problem.log['I_AMPA_ext'])
plt.plot(IAMPAext_hist, color='darkblue')

#%%
sAMPA_hist = np.array(wang_nc.problem.log['s_AMPA'])
sAMPA_histE = np.array(wang_nc.problem.log['s_AMPA_E'])
IAMPA_hist = np.array(wang_nc.problem.log['I_AMPA'])
IAMPA_histE = np.array(wang_nc.problem.log['I_AMPA_E'])

plt.figure()
plt.plot(sAMPA_hist, color='darkblue')
plt.plot(sAMPA_histE, color='darkred')
plt.title('s AMPA')
plt.ylim([0, 6.2e-8])
plt.grid()

plt.figure()
plt.plot(IAMPA_histE, color='darkred')
plt.plot(IAMPA_hist, color='darkblue')
plt.title('I AMPA')
plt.ylim([-3.5e-9, 0])
plt.grid()

#%%
vHist = np.array(wang_nc.problem.log['V'])
plt.figure()
plt.plot(vHist, color='darkred')
plt.title('V')

print("Max Rate for Group I: {}".format(rate_I.max()))

# xNMDA_hist = np.array(wang_nc.problem.x_NMDA_hist)
# plt.figure()
# plt.plot(xNMDA_hist[:, 125])
# plt.plot(xNMDA_hist[:, 310])
# plt.plot(xNMDA_hist[:, 1825])

# #%%
# sNMDA_hist = np.array(wang_nc.problem.s_NMDA_hist)
# # plt.figure()
# plt.plot(sNMDA_hist[:, 125])
# plt.plot(sNMDA_hist[:, 310])
# plt.plot(sNMDA_hist[:, 1825])

# #%%
# INMDA_hist = np.array(wang_nc.problem.I_AMPA_hist)
# plt.figure()
# # plt.plot(INMDA_hist[:, 125])
# # plt.plot(INMDA_hist[:, 310])
# plt.plot(INMDA_hist[:, 1580])