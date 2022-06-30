#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 21:11:32 2022

@author: aggelen
"""

from spayk.Core import Simulator
from spayk.Stimuli import ExternalCurrentSignal, SpikeTrains
from spayk.Organization import Tissue
from spayk.Nerves import SpikeResponseLIF

import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.close('all')

T = 1000
dt = 1.0
n_syn = 500

spike_trains = SpikeTrains(n_syn, dt=dt)

stimulus = spike_trains.add_spikes(int(T/dt))

W = np.full((n_syn), 0.475, dtype=np.float32)

params = {'n_syn': n_syn,
          'w': W,
          'v_rest': 0.0,
          'tau_rest': 1.0, 
          'tau_m': 10.0, 
          'tau_s': 2.5, 
          'K': 2.1, 
          'K1': 2.0, 
          'K2': 4.0,
          'dt': dt}

network = SpikeResponseLIF(params)

sim_params = {'dt': dt,
              't_stop': T,
              'stimuli': stimulus}

sim0 = Simulator()
sim0.integrate_and_fire(network, sim_params)

    
plt.figure()
plt.plot(np.arange(sim0.results['v_out'].__len__())*dt, np.array(sim0.results['v_out']))
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential')
plt.title('Membrane Potential')
