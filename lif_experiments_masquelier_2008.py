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

T = 15000
dt = 1.0
steps = int(T / dt)
n_syn = 2000
n_syn_pattern = int(n_syn/2)
pattern_t = []

spike_trains = SpikeTrains(n_syn, delta_max=50, dt=dt)
spike_trains.add_spikes(T)

spike_trains_c = SpikeTrains(n_syn, r=np.full((n_syn), 10), auto_vrate=False, dt=dt)
spike_trains_c.add_spikes(T)

syn_has_spiked = np.zeros((steps, n_syn), dtype=np.bool)

# beginning of the pattern in the [25,75] ms interval
pat_start_time = np.random.randint(25,75)
pattern_t.append(pat_start_time)

for step in range(steps):
    t = int(step * dt)

    syn_has_spiked[step,:] = spike_trains.spikes[step,:]

    if t >= pat_start_time and t < (pat_start_time + 50):
        syn_has_spiked[step,:n_syn_pattern] = spike_trains.spikes[t - pat_start_time + pattern_t[0],:n_syn_pattern]
    else:
        if t >= pat_start_time + 100:
            pat_start_time = t
            # We have 1/4 chances of replaying the pattern for each chunk of 50 ms
            r = np.random.uniform(0,1)
            while (r >= 0.25):
                pat_start_time += 50
                r = np.random.uniform(0,1)
            pattern_t.append(pat_start_time)
    syn_has_spiked[step,:] |= spike_trains_c.spikes[step,:]

#%%
plt.rcParams["figure.figsize"] =(15,6)
# Evaluate the mean firing rate of each synapse in Hz
rates = np.count_nonzero(syn_has_spiked, axis=0)*1000.0/T
r_max = np.max(rates)
r_mean_a = np.mean(rates[:n_syn_pattern])
r_mean_b = np.mean(rates[n_syn_pattern:])
plt.figure()
plt.title('Synapse mean firing rates')
plt.plot(rates)
plt.axhline(y=r_mean_a, xmax=0.5, color='g', linestyle='--')
plt.text(0,r_max -10,'mean rate: %d' % r_mean_a, color='g')
plt.axhline(y=r_mean_b, xmin=0.5, color='r', linestyle='--')
plt.text(n_syn,r_max -10,'mean rate: %d' % r_mean_b, ha='right', color='r')
intervals = ([0,299],[7200,7499], [14700, 14999])
for interval in intervals:
    it_pattern_t = np.array(pattern_t)
    it_pattern_t = it_pattern_t[np.logical_and(it_pattern_t >=interval[0], it_pattern_t <=interval[1])]
    # Draw input spikes, identifying the patterns
    plt.figure()
    it_spikes = syn_has_spiked[interval[0]:interval[1]]
    it_real_spikes = np.argwhere(it_spikes)
    for pat_t in it_pattern_t:
        plt.fill_between((pat_t,np.minimum(interval[1],pat_t+50),np.minimum(interval[1],pat_t+50),pat_t),
                         (0,0,n_syn_pattern,n_syn_pattern),facecolor='lightgray')
    t, s = it_real_spikes.T
    plt.scatter(interval[0] + t,s+1,s=1)

W = np.full((n_syn), 0.475, dtype=np.float32)

stdp_params = {'a_plus': 0.03125, 'a_minus': 0.029, 'tau_plus': 16.8, 'tau_minus': 33.7}

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
          'stdp_params': stdp_params}

network = STDPSpikeResponseLIF(params)

sim_params = {'dt': dt,
              't_stop': T,
              'stimuli': syn_has_spiked}

sim0 = Simulator()
sim0.integrate_and_fire_stdp(network, sim_params)

    
# plt.figure()
# plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['v_out']))
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential')
# plt.title('Membrane Potential')

print('Firing Rate: {}'.format(np.sum(np.array(sim0.results['v_out'])> network.v_th)))

#%%

P = np.array(sim0.results['v_out'])
t = np.arange(P.__len__())

P = np.c_[t,P]

plt.rcParams["figure.figsize"] =(15,3)
intervals = ([0,1999],[4000,5999], [8000,9999], [13000, 14999])

for interval in intervals:
    it_P = P[interval[0]:interval[1]]
    it_pattern_t = np.array(pattern_t)
    it_pattern_t = it_pattern_t[np.logical_and(it_pattern_t >= interval[0], it_pattern_t <=interval[1])]
    # Draw membrane potential, identifying the patterns
    plt.figure()
    for pat_t in it_pattern_t:
        plt.fill_between((pat_t,pat_t+50,pat_t+50,pat_t),(-network.v_th/2,-network.v_th/2,network.v_th*2,network.v_th*2),facecolor='lightgray')
    plt.plot(it_P[:,0], it_P[:,1])
    plt.axhline(y=network.v_th, color='r', linestyle='-')
    plt.axhline(y=network.v_rest, color='y', linestyle='--')
    plt.title('LIF response')
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (msec)')

# plt.figure()
# plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['v_out']))
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential (mV)')
# plt.title('Membrane Potential')

# plt.figure()
# plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['mean_w']))
# plt.xlabel('Time (ms)')
# plt.ylabel('Mean Weights')
# plt.title('STDP Mean Weight Change')
