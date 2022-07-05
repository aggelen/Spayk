#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:47:17 2022

@author: aggelen
"""

import sys
sys.path.append('..')

from spayk.Organization import Tissue
from spayk.Models import SRMLIFNeuron
from spayk.Stimuli import ExternalSpikeTrain

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

T = 100
dt = 1.0
steps = int(T / dt)
m = 5
spikes = np.zeros((steps,m), dtype=np.bool)
spike_times = np.array([2, 23, 34, 35, 38, 51, 60, 62, 65, 66, 67, 75])
spike_index = np.array([1,  3,  2,  0,  4,  1,  3,  2,  3,  0,  1,  1])
spikes[spike_times, spike_index] = True

W = np.full((m), 0.475, dtype=np.float32)

stdp_params = {'a_plus': 0.03125, 'a_minus': 0.029, 'tau_plus': 16.8, 'tau_minus': 33.7}
# create two neurons as a group with LIF/SRM model
group_params = {'n_synapses': 5,
                'dt': 1.0,
                'v_th': 1.0,
                'w': W,
                'stdp_on': True,
                'stdp_params': stdp_params}

srm_neuron_group = SRMLIFNeuron(group_params)

# bind neuron groups to a tissue
test_tissue = Tissue([srm_neuron_group])
input_spike_train = ExternalSpikeTrain(dt=1.0, t_stop=T, no_neurons=5, spike_train=spikes.T)
# input_spike_train.raster_plot()

# run simulation
test_tissue.keep_alive(stimuli=input_spike_train)
test_tissue.logger.plot_v()
# test_tissue.logger.raster_plot()