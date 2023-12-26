#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:49:01 2023

@author: gelenag
"""

from spayk.Models import SRMLIFNeuron, IzhikevichNeuronGroup
from spayk.Stimuli import SpikeInstance
from spayk.Organization import Tissue
from spayk.Learning import OnlineSTDP

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

robot_spikes = np.load("/home/gelenag/Dev/CA1_Robot/ca1_robot_spikes.npy")

# positions = [np.argwhere(p)[:,0].tolist() for p in robot_spikes.T]
# plt.figure("Raster Plot")
# plt.eventplot(positions)
# plt.grid()


stim_freq = 30 # Hz
neuron = SRMLIFNeuron({'n_synapses': 9, 'dt': 1/stim_freq})

group_params = {'no_neurons': 1,
                'behaviour': 'synaptic',
                'no_syn': 9,
                'E': np.zeros(9)}

neuron = IzhikevichNeuronGroup(group_params)
neuron.autoconnect()

stim = SpikeInstance(dt=1/stim_freq)
ca1 = Tissue([neuron])

stdp_params = {'no_presyn_neurons': 9, 
               'no_postsyn_neurons': 1, 
               'dt': 1/stim_freq, 
               'tau_pre': 1.2*(1/stim_freq), 
               'tau_post': 1*(1/stim_freq),
               'eta_post': 0.001,
               'eta_pre': 0.01}

plasticity = OnlineSTDP(stdp_params)



# new_weights = plasticity.offline(pre_spikes=input_spike_train.spikes, 
#                                  post_spikes=sup_sig.spikes, 
#                                  weights=neurongroup.w,
#                                  learning_rate=5e-3,
#                                  log_changes=True)

op = []
output_spikes = []

dw_log_pre = []
dw_log_post = []

for spikes in robot_spikes:
    stim.load_spikes(spikes)
    ca1.keep_alive(stim)
    
    plasticity.presynaptic_trace_updates(spikes)

    post_syn_spikes = ca1.get_spikes()
    output_spikes.append(post_syn_spikes)

    plasticity.postsynaptic_trace_updates(post_syn_spikes)

    op.append(ca1.memb_pot)
    
    dw_presyn, dw_postsyn = plasticity.weight_change(ca1.neuron_group.w)
    dw_log_pre.append(dw_presyn.flatten())
    dw_log_post.append(dw_postsyn.flatten()) 
    
op = np.array(op)

#%%
plt.figure("Memb. Pot.")
plt.plot(op)

#%%
plt.figure("Presyn_trace_log")
presyn_traces = np.array(plasticity.presyn_trace_log)
plt.plot(presyn_traces[:,1])

plt.figure("Postsyn_trace_log")
postsyn_traces = np.array(plasticity.postsyn_trace_log)
plt.plot(postsyn_traces[:,0])

#%%
plt.figure("dw_log")
dws_pre = np.array(dw_log_pre)
dws_post = np.array(dw_log_post)
# plt.plot(dws_pre[:,0])
# plt.plot(dws_post[:,0])

positions = [np.argwhere(p)[:,0].tolist() for p in dws_pre.T]
plt.figure("Raster Plot Pre")
plt.eventplot(positions)
plt.grid()

positions = [np.argwhere(p)[:,0].tolist() for p in dws_post.T]
plt.figure("Raster Plot Post")
plt.eventplot(positions)
plt.grid()