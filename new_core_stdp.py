#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:03:39 2022

@author: aggelen
"""

from spayk.Nerves import STDPIzhikevichNeuronGroup
from spayk.Core import Simulator
from spayk.Stimuli import SpikeTrains

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#%% Stim

T = 100
dt = 0.1
steps = int(T / dt)

# Number of synapses
m = 5

spikes = np.zeros((steps,m))

spike_times = np.array([20, 230, 340, 350, 380, 510, 600, 620, 650, 660, 670, 750])
spike_index = np.array([1,  3,  2,  0,  4,  1,  3,  2,  3,  0,  1,  1])
spikes = spikes.astype(bool)
spikes[spike_times, spike_index] = True

# %% 3. Create Model
n = 1
m = 5

w_in = 0.05*np.ones((n,m), dtype=np.float32)

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

#%% Visualize

# v_out = sim0.results['v_out']


#%%
delta_weights = sim0.results['delta_weights']
rewards = np.argwhere(delta_weights > 0)
rewards_timings = rewards[:,0]
rewards_index = rewards[:,1] + 1
penalties = np.argwhere(delta_weights < 0)
penalties_timings = penalties[:,0]
penalties_index = penalties[:,1] + 1

plt.figure()
plt.axis([0, 1050, 0, 6])
plt.title('Synaptic spikes')
plt.ylabel('synapses')
plt.xlabel('Time (msec)')
plt.scatter(spike_times, spike_index+1, s=100)
for spike in spike_times:
    plt.axvline(x=spike, color='gray', linestyle='--')
    

plt.scatter(rewards_timings, rewards_index, color='lightgreen')
plt.scatter(penalties_timings, penalties_index, color='red')
# Draw membrane potential
plt.figure()
for spike in spike_times:
    plt.axvline(x=spike, color='gray', linestyle='--')
    
plt.axis([0, 1050, -90, 50])

# plt.axhline(y=100, color='r', linestyle='-')
# plt.axhline(y=tissuenetworkp_rest, color='y', linestyle='--')
plt.plot(sim0.results['v_out'])
# plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')


plt.figure()
plt.plot(sim0.results['v_out'])
for spike in spike_times:
    plt.axvline(x=spike, color='gray', linestyle='--')
# plt.axis([0, 1050, 0, 80])    
    
plt.figure()
plt.plot(sim0.results['I_in'])
for spike in spike_times:
    plt.axvline(x=spike, color='gray', linestyle='--')
# plt.axis([0, 1050, 0, 6])    