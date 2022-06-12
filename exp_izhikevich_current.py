#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:28:25 2022

@author: aggelen
"""

from spayk.Nerves import SynapticIzhikevichNeuronGroup, STDPIzhikevichNeuronGroup
from spayk.Core import Simulator
from spayk.Stimuli import SpikeTrains

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.close('all')

res_firing_rate = []
res_mean_currents = []
for epoch in tqdm(range(100)):
    T = 1000
    dt = 0.1
    no_neurons = 1
    n_syn = 5

    spike_trains = SpikeTrains(n_syn, r_min=20, r_max=40)
    stimuli = spike_trains.add_spikes(int(T/dt))

    w_in = 0.6*np.ones((no_neurons,n_syn), dtype=np.float32)

    #%% 2. Create Model
    params = {'no_neurons': no_neurons,
              'dynamics': 'regular_spiking',
              'no_connected_neurons': n_syn,
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
    # plt.plot(np.arange(len(sim0.results['v_out']))*0.1, sim0.results['v_out'])
    
    # plt.figure('current')
    # plt.plot(np.arange(len(sim0.results['I_in']))*0.1, sim0.results['I_in'])
    
    # plt.figure()
    # presyn_spikes = np.array(sim0.results['presyn_spikes'])
    # # spike_times = np.where(presyn_spikes)[0]*0.1
    
    # sc_spikes = np.argwhere(presyn_spikes)
    
    # steps, neurons = sc_spikes.T
    # plt.scatter(steps*0.1, neurons, s=3)
    # plt.xlim([0,1000])
     
    # neuron_idx = np.arange(presyn_spikes.shape[1])
    # plt.scatter(spike_times, neuron_idx)
    # plt.eventplot(spike_times)
    # plt.xlim([0,1000])
    res_firing_rate.append(sum(np.array(sim0.results['v_out'])>34))
    # print('Resulting Firing Rate: {}'.format(sum(np.array(sim0.results['v_out'])>34)))
    res_mean_currents.append(np.mean(sim0.results['I_in']))

#%%
print('Mean Firing Rate: {}'.format(np.mean(res_firing_rate)))
print('Mean Current: {}'.format(np.mean(res_mean_currents)))