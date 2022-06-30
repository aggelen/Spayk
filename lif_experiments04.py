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

T = 1000
dt = 1.0
steps = int(T / dt)

n_syn = 500
# Spike trains
spike_trains = [SpikeTrains(n_syn, r_max=70, dt=dt),
                SpikeTrains(n_syn, r_max=90, dt=dt),
                SpikeTrains(n_syn, r_max=110, dt=dt)]

# spike_train = SpikeTrains(n_syn, r_max=20)
# st = spike_train.add_spikes(int(T/dt))
st = spike_trains[0].add_spikes(int(T/dt))
# st = spike_trains[1].add_spikes(int(T/dt))
# st = spike_trains[2].add_spikes(int(T/dt))

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
          'stdp_params': stdp_params}

network = STDPSpikeResponseLIF(params)

sim_params = {'dt': dt,
              't_stop': T,
              'stimuli': st}

sim0 = Simulator()
sim0.integrate_and_fire_stdp(network, sim_params)

    
# plt.figure()
# plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['v_out']))
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential')
# plt.title('Membrane Potential')

print('Firing Rate: {}'.format(np.sum(np.array(sim0.results['v_out'])> network.v_th)))

plt.figure()
plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['v_out']))
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potential')

plt.figure()
plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['mean_w']))
plt.xlabel('Time (ms)')
plt.ylabel('Mean Weights')
plt.title('STDP Mean Weight Change')
