#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 21:11:32 2022

@author: aggelen
"""

from spayk.Core import Simulator
from spayk.Stimuli import ExternalCurrentSignal, SpikeTrains
from spayk.Organization import Tissue
from spayk.Nerves import STDPSpikeResponseLIF

import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.close('all')

T = 100
dt = 1.0
steps = int(T / dt)
n_syn = 5

spikes = np.zeros((steps,n_syn), dtype=np.bool)
spike_times = np.array([2, 23, 34, 35, 38, 51, 60, 62, 65, 66, 67, 75])
spike_index = np.array([1,  3,  2,  0,  4,  1,  3,  2,  3,  0,  1,  1])
spikes[spike_times, spike_index] = True

W = np.full((n_syn), 0.475, dtype=np.float32)

stdp_params = {'a_plus': 0.03125, 'a_minus': 0.0265625, 'tau_plus': 16.8, 'tau_minus': 33.7}

params = {'n_syn': n_syn,
          'w': W,
          'v_rest': 0.0,
          'tau_rest': 1.0, 
          'tau_m': 10.0, 
          'tau_s': 2.5, 
          'K': 2.1, 
          'K1': 2.0, 
          'K2': 4.0,
          'dt': dt,
          'v_th': 1.0,
          'stdp_params': stdp_params}

network = STDPSpikeResponseLIF(params)

sim_params = {'dt': dt,
              't_stop': T,
              'stimuli': spikes}

sim0 = Simulator()
sim0.integrate_and_fire_stdp(network, sim_params)

    
# plt.figure()
# plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['v_out']))
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential')
# plt.title('Membrane Potential')

print('Firing Rate: {}'.format(np.sum(np.array(sim0.results['v_out'])> network.v_th)))

delta_weights = np.array(sim0.results['delta_w'])
rewards = np.argwhere(delta_weights > 0)
rewards_timings = rewards[:,0]
rewards_index = rewards[:,1] + 1
penalties = np.argwhere(delta_weights < 0)
penalties_timings = penalties[:,0]
penalties_index = penalties[:,1] + 1
plt.figure()
plt.axis([0, T, 0, network.n_syn + 1])
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
plt.axhline(y=network.v_th, color='r', linestyle='-')
plt.axhline(y=network.v_rest, color='y', linestyle='--')

plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['v_out']))
plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')
