#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:03:39 2022

@author: aggelen
"""

from spayk.Nerves import SynapticIzhikevichNeuronGroup, STDPIzhikevichNeuronGroup
from spayk.Core import Simulator
from spayk.Stimuli import SpikeTrains

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
#%% Stim
# n_syn = 50
# T = 2000
# test_a = SpikeTrains(n_syn)
# test_a.add_spikes(T)
# test_b = SpikeTrains(n_syn, delta_max=50)
# test_b.add_spikes(T)

# for test in (test_a, test_b):
#     # Draw input spikes
#     plt.figure()
#     plt.axis([0, T, 0, test.n_syn])
#     # Evaluate the mean firing rate
#     rate = np.count_nonzero(test.spikes)*1000.0/T/test.n_syn
#     plt.title('Synaptic spikes with varying rates [%d,%d] Hz. Resulting mean rate: %d Hz' % (test.r_min, test.r_max, rate))
#     plt.ylabel('synapses')
#     plt.xlabel('Time (msec)')
#     t, spikes = test.get_spikes()
#     plt.scatter(t, spikes, s=2)

T = 1000
dt = 0.1
n_syn = 500
spike_trains = SpikeTrains(n_syn)
stimuli = spike_trains.add_spikes(int(T/dt))

# %% 3. Create Model
n = 10
m = 500

w_in = 0.7*np.ones((n,m), dtype=np.float32)

params = {'no_neurons': 1,
          'dynamics': 'regular_spiking',
          'no_connected_neurons': m,
          'tau': 10.0,
          'W_in': w_in,
          'dt': 0.1}

network = SynapticIzhikevichNeuronGroup(params)

#%% Simulate
sim_params = {'dt': 0.1,
              't_stop': 1000,
              'frate': 0.002,
              'stimuli': stimuli}

sim0 = Simulator()
sim0.new_core_syn(network, sim_params)

#%% Visualize

# v_out = sim0.results['v_out']
plt.figure()
plt.plot(sim0.results['v_out'])
