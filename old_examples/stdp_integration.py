#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:32:58 2022

@author: aggelen
"""
import sys
sys.path.append("..") 

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import time

from spayk.Generators import PoissonSpikeGenerator
from spayk.Visualization import raster_plot, plot_voltage_traces

from spayk.Synapses import STDP

#%% STDP Demo
tau_stdp = 20
stdp_prototype = STDP(A_plus=0.008, A_minus=0.008*1.10, tau_stdp=tau_stdp)
time_diff = np.linspace(-5 * tau_stdp, 5 * tau_stdp, 50)
dW = stdp_prototype.calculate_dW(time_diff)

plt.figure('STDP Demo')
plt.plot(time_diff, dW)
plt.xlabel('Time Difference (ms)')
plt.ylabel('Change in Synaptic Strength')
plt.grid()

#%% Scenario: 5 presyn -> 1 postsyn
pg = PoissonSpikeGenerator(dt=0.1)
presyn_spike_trains = pg.generate(t_end=1000, no_neurons=300, firing_rate=10)
raster_plot(presyn_spike_trains, dt=0.1)

stdp0 = STDP(A_plus=0.008, A_minus=0.008*1.10, tau_stdp=20)
LTPs = stdp0.calculate_LTP(presyn_spike_trains, dt=0.1)
plot_voltage_traces(LTPs, selected=[1,3,5,7,9])

v, post_syn_spike_train, gE, LTP, LTD, d_gE = stdp0.update_weights(presyn_spike_trains, dt=0.1)
plot_voltage_traces(np.array([v, gE, LTD]))

plot_voltage_traces(d_gE, selected=[1,3,5,7,9])