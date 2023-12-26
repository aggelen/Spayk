#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:44:28 2023

@author: gelenag

Single SRM neuron connected to the 9 visual stim. neuron with plastic synapses
Data recorded from Webots
"""

from spayk.Models import SRMLIFNeuron
from spayk.Stimuli import SpikeInstance
from spayk.Organization import Tissue
from spayk.Learning import OnlineSTDP

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

robot_spikes = np.load("/home/gelenag/Dev/CA1_Robot/ca1_robot_spikes.npy")

positions = [np.argwhere(p)[:,0].tolist() for p in robot_spikes.T]
plt.figure("Raster Plot")
plt.eventplot(positions)
plt.grid()

stim_freq = 30 # Hz
neuron = SRMLIFNeuron({'n_synapses': 9, 'dt': 1/stim_freq})
w_initial = neuron.w

stim = SpikeInstance(dt=1/stim_freq)
ca1 = Tissue([neuron])

stdp_params = {'no_presyn_neurons': 9, 
                'no_postsyn_neurons': 1, 
                'dt': 1/stim_freq, 
                'tau_pre': 1.9*(1/stim_freq), 
                'tau_post': 1.6*(1/stim_freq),
                'lr_ltp': 0.05,
                'lr_ltd': 1.0}
plasticity = OnlineSTDP(stdp_params)

#%% Without STDP, Static Synapses
neuron_output = []
for spikes in robot_spikes:
    stim.load_spikes(spikes)
    ca1.keep_alive(stim)
    neuron_output.append(ca1.memb_pot)
    
neuron_output = np.array(neuron_output)
plt.figure("Memb. Pot. wo STDP")
plt.plot(neuron_output)

#%% With STDP, Plastic Synapses
neuron = SRMLIFNeuron({'n_synapses': 9, 'dt': 1/stim_freq})
neuron.w = w_initial
stim = SpikeInstance(dt=1/stim_freq)
ca1 = Tissue([neuron])

output_spikes = []
neuron_output = []
ltp_log = []
ltd_log = []
w_log = []
for spikes in robot_spikes:
    stim.load_spikes(spikes)
    ca1.keep_alive(stim)
    neuron_output.append(ca1.memb_pot)
    
    postsyn_spikes = ca1.get_spikes()
    output_spikes.append(postsyn_spikes)
    
    plasticity.presynaptic_trace_updates(spikes)
    plasticity.postsynaptic_trace_updates(postsyn_spikes)

    ltp = plasticity.LTP(postsyn_spikes)
    ltp_log.append(ltp.flatten())
    
    ltd = plasticity.LTP(spikes)
    ltd_log.append(ltd.flatten())
    
    dw = ltp - ltd
    neuron.w = neuron.w + dw
    w_log.append(neuron.w.flatten())
    
    
neuron_output = np.array(neuron_output)
plt.figure("Memb. Pot. with STDP")
plt.plot(neuron_output)

#%%
i = 0 # which synapse
plt.figure()
presyn_traces = np.array(plasticity.presyn_trace_log)
plt.plot(presyn_traces[:,i])

postsyn_traces = np.array(plasticity.postsyn_trace_log)
plt.plot(postsyn_traces[:,i])

#%%
i = 0 # which synapse
plt.figure()
w_log = np.array(w_log)
plt.plot(w_log)
