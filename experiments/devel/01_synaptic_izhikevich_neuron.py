#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:55:11 2022

@author: aggelen
"""

import sys
sys.path.append('../..')

from spayk.Organization import Tissue
from spayk.Models import IzhikevichNeuronGroup
from spayk.Stimuli import PoissonSpikeTrain

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# stimuli
no_syn = 5
no_inh = 2
s_rts = 10*np.ones(no_syn)
inh_neuron_idx = np.random.choice(no_syn, size=no_inh, replace=False)
s_rts[inh_neuron_idx] = np.random.choice(np.arange(30,45), size=no_inh)
input_spike_train = PoissonSpikeTrain(dt=0.1, t_stop=1000, no_neurons=no_syn, spike_rates=10*np.ones(no_syn))
neuron_types = np.zeros(no_syn)
neuron_types[inh_neuron_idx] = 1
input_spike_train.raster_plot(color_array=neuron_types)

#%% Synaptic Model
no_neurons = 1
E = np.zeros(no_syn)
E[inh_neuron_idx] = -75

group_params = {'no_neurons': no_neurons,
                'behaviour': 'synaptic',
                'no_syn': no_syn,
                'E': E}

neurongroup1 = IzhikevichNeuronGroup(group_params)
neurongroup1.autoconnect()

tissue = Tissue([neurongroup1])

tissue.keep_alive(stimuli=input_spike_train)
tissue.logger.plot_v()
