#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 21:23:21 2022

@author: aggelen
"""

from spayk.Core import Simulator
from spayk.Stimuli import ExternalCurrentSignal, SpikeTrains
from spayk.Organization import Tissue
from spayk.Nerves import SynapticIzhikevichNeuronGroup, STDPIzhikevichNeuronGroup
from spayk.Synapses import Synapse, GENESIS_Synapse

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

dt = 0.1
m = 5
steps = 1000
spikes = np.zeros((steps,m))

spike_times = np.array([20, 230, 340, 350, 380, 510, 600, 620, 650, 660, 670, 750])
spike_index = np.array([1,  3,  2,  0,  4,  1,  3,  2,  3,  0,  1,  1])
spikes = spikes.astype(bool)
spikes[spike_times, spike_index] = True

plt.figure()
plt.axis([0, steps, 0, m+2])
steps, neurons = np.argwhere(spikes).T
plt.scatter(steps, neurons+1, s=3)

#%% Network

w_in = 0.6*np.ones((1,m), dtype=np.float32)

params = {'no_neurons': 1,
          'dynamics': 'regular_spiking',
          'no_connected_neurons': m,
          'tau': 8.0,
          'W_in': w_in,
          'dt': dt,
          'a_plus': 0.03125,
          'a_minus': 0.0265625,
          'tau_plus': 16.8,
          'tau_minus': 33.7}

network = STDPIzhikevichNeuronGroup(params)

#%% Simulate
sim_params = {'dt': 0.1,
              't_stop': 100,
              'frate': 0.002,
              'stimuli': spikes}

sim0 = Simulator()
sim0.new_core_syn_stdp(network, sim_params)

#%%
v_out = sim0.results['v_out']

out_spikes = np.array(v_out) > 34.9
steps, neurons = np.argwhere(out_spikes).T
plt.scatter(steps, 6*np.ones_like(steps), s=3)


delta_weights = sim0.results['delta_weights']
rewards = np.argwhere(delta_weights > 0)
rewards_timings = rewards[:,0]
rewards_index = rewards[:,1] + 1
penalties = np.argwhere(delta_weights < 0)
penalties_timings = penalties[:,0]
penalties_index = penalties[:,1] + 1

plt.scatter(rewards_timings, rewards_index, color='lightgreen')
plt.scatter(penalties_timings, penalties_index, color='red')

plt.figure()
plt.plot(v_out)