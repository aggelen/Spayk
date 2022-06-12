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
# Duration of the simulation in ms
T = 1000
# Duration of each time step in ms
dt = 0.1
# Number of iterations = T/dt
steps = int(T / dt)
# Number of synapses
n_syn = 500
# Spike trains
st = [SpikeTrains(n_syn, r_max=70),
                SpikeTrains(n_syn),
                SpikeTrains(n_syn, r_max=110)]

n = 1
m = n_syn
spike_trains = []
for i in range(3):
    spike_trains.append(st[i].add_spikes(T))
    
    

sc_spikes = np.argwhere(spike_trains[0])

steps, neurons = sc_spikes.T
plt.figure()
plt.scatter(steps*dt, neurons, s=3)


w_in = 0.6*np.ones((n,m), dtype=np.float32)

params = {'no_neurons': 1,
          'dynamics': 'regular_spiking',
          'no_connected_neurons': m,
          'tau': 10.0,
          'W_in': w_in,
          'dt': dt,
          'a_plus': 0.03125,
          'a_minus': 0.0265625,
          'tau_plus': 10,
          'tau_minus': 20}

network = STDPIzhikevichNeuronGroup(params)

#%% Simulate
sim_params = {'dt': dt,
              't_stop': 100,
              'frate': 0.002,
              'stimuli': spike_trains[0]}

sim0 = Simulator()
sim0.new_core_syn_stdp(network, sim_params)


#%%
plt.figure()
plt.plot(sim0.results['v_out'])

plt.figure()
plt.plot(sim0.results['I_in'])

plt.figure()
plt.plot(sim0.results['delta_weights'])