#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:44:28 2023

@author: gelenag
"""

from spayk.Models import SRMLIFNeuron, IzhikevichNeuronGroup
from spayk.Stimuli import SpikeInstance
from spayk.Organization import Tissue
from spayk.Learning import OnlineSTDP

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

presyn_spikes = np.zeros(60)
presyn_spikes[[5,17,35,55]] = 1

postsyn_spikes = np.zeros(60)
postsyn_spikes[[10, 20,45,50]] = 1

pre_times = np.argwhere(presyn_spikes).flatten()
post_times = np.argwhere(postsyn_spikes).flatten()
positions = [post_times, pre_times]

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
ax1.eventplot(positions)
ax1.set_ylabel("pre/post syn spikes")


stim_freq = 30 # Hz

stdp_params = {'no_presyn_neurons': 1, 
                'no_postsyn_neurons': 1, 
                'dt': 1/stim_freq, 
                'tau_pre': 5*(1/stim_freq), 
                'tau_post': 5*(1/stim_freq),
                'eta_post': 0.01,
                'eta_pre': 0.01}

plasticity = OnlineSTDP(stdp_params)

w = np.random.uniform(0.3, 0.7, (1, 1)) 

dw_log = []
dw_log_pre = []
dw_log_post = []
ltp_log, ltd_log = [], []
w_log = []
for ti, spikes in enumerate(presyn_spikes):
    plasticity.presynaptic_trace_updates(spikes)
    plasticity.postsynaptic_trace_updates(postsyn_spikes[ti])

    ltp = plasticity.LTP(postsyn_spikes[ti])
    ltp_log.append(ltp)
    
    ltd = plasticity.LTP(spikes)
    ltd_log.append(ltd)
    
    dw = ltp - ltd
    w = w + dw
    w_log.append(w.flatten())
    
    # dw_presyn, dw_postsyn = plasticity.weight_change(w)
    # dw_log_pre.append(dw_presyn.flatten())
    # dw_log_post.append(dw_postsyn.flatten()) 
    # dw_log.append((dw_postsyn-dw_presyn).flatten())
    

#%%
presyn_traces = np.array(plasticity.presyn_trace_log)
ax2.plot(presyn_traces)
ax2.set_ylabel("presyn traces")

postsyn_traces = np.array(plasticity.postsyn_trace_log)
ax3.plot(postsyn_traces)
ax3.set_ylabel("postsyn traces")


# #%%
ltp_log = np.array(ltp_log)
ltd_log = np.array(ltd_log)
ax4.plot(ltp_log)
ax4.plot(ltd_log)
ax4.set_ylabel("ltp/ltd signal")

w_log = np.array(w_log)
ax5.plot(w_log)
ax5.set_ylabel("weight change")

# plt.plot(dws_pre[:,0])
# plt.plot(dws_post[:,0])

# positions = [np.argwhere(p)[:,0].tolist() for p in dws_pre.T]
# plt.figure("Raster Plot Pre")
# plt.eventplot(positions)
# plt.grid()

# positions = [np.argwhere(p)[:,0].tolist() for p in dws_post.T]
# plt.figure("Raster Plot Post")
# plt.eventplot(positions)
# plt.grid()