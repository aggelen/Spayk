#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 22:32:59 2022

@author: aggelen
"""
from spayk.Core import Simulator
from spayk.Stimuli import ExternalCurrentSignal, SpikeTrains
from spayk.Organization import Tissue
from spayk.Nerves import STDPIzhikevichNeuronGroup, STDPRecurrentIzhikevichNeuronGroup, RecurrentIzhikevichNeuronGroup
from spayk.Synapses import Synapse, GENESIS_Synapse
# from Spayk.Observation import ObservationSettings

import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.close('all')

T = 1000
dt = 0.1

n_syn = 500
# Spike trains
spike_trains = [SpikeTrains(n_syn, r_max=70),
                SpikeTrains(n_syn),
                SpikeTrains(n_syn, r_max=110)]

# spike_train = SpikeTrains(n_syn, r_max=20)
# st = spike_train.add_spikes(int(T/dt))
# st35 = spike_trains[0].add_spikes(int(T/dt))
st = spike_trains[0].add_spikes(int(T/dt))
# st55 = spike_trains[2].add_spikes(int(T/dt))

n = 1
m = n_syn

w_in = np.full((n,m), 0.475, dtype=np.float32)

params = {'no_neurons': n,
          'dynamics': 'regular_spiking',
          'no_connected_neurons': m,
          'tau': 10.0,
          'W_in': w_in,
          'dt': 0.1,
          'a_plus': 0.03125,
          'a_minus': 0.0265625,
          'tau_plus': 16.8,
          'tau_minus': 33.7}


network = STDPIzhikevichNeuronGroup(params)

#%% Simulate
t_stop = 1000
sim_params = {'dt': 0.1,
              't_stop': t_stop,
              'frate': 0.002,
              'stimuli': st}

sim0 = Simulator()
sim0.new_core_syn_experimental(network, sim_params)

#%%
plt.figure()
plt.plot(np.array(sim0.results['v_out']))

plt.figure()
plt.plot(np.array(sim0.results['weight_means']))


#%%
# meanfr = np.sum(st35,0).mean()
# sc_spikes = np.argwhere(st35)
# steps, neurons = sc_spikes.T
# plt.figure()
# plt.scatter(steps*0.1, neurons, s=3)
# sc_spikes = np.argwhere(st45)
# steps, neurons = sc_spikes.T
# plt.figure()
# plt.scatter(steps*0.1, neurons, s=3)
# sc_spikes = np.argwhere(st55)
# steps, neurons = sc_spikes.T
# plt.figure()
# plt.scatter(steps*0.1, neurons, s=3)