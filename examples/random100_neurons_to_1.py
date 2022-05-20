#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:00:25 2022

@author: aggelen
"""
import sys
sys.path.append("..") 

from spayk.Core import Simulator
from spayk.Stimuli import ConstantCurrentSource
from spayk.Organization import Tissue
from spayk.Nerves import SingleIzhikevichNeuron
from spayk.Synapses import Synapse, GENESIS_Synapse
# from Spayk.Observation import ObservationSettings

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
#%% Create Network
tissue = Tissue()

#%% 100 presyn neurons connected to 1 postsyn neuron with random weights
presyn_neurons = []
for i in range(100):
    n = SingleIzhikevichNeuron(stimuli=ConstantCurrentSource(np.random.randint(5,45)))
    presyn_neurons.append(n)
tissue.add(presyn_neurons)

input_neuron_idx = np.arange(100)
output_neuron_idx = max(input_neuron_idx) + 1

gs_01 = GENESIS_Synapse(params={'tau': 10, 'dt': 0.1}, io_neuron_idx=[input_neuron_idx, output_neuron_idx])
n_postsyn = SingleIzhikevichNeuron(stimuli=gs_01)
tissue.add([n_postsyn])

#%% Embody
tissue.embody()

#%% Observation Settings
settings = {'duration': 1000,
            'dt': 0.1}

#%% Run
sim0 = Simulator()
sim0.keep_alive(tissue, settings)

#%% Aux.

tissue.raster_plot(dt=0.1)

tissue.plot_membrane_potential_of(output_neuron_idx, dt=0.1, color='g')
tissue.plot_current_of_neuron(output_neuron_idx, dt=0.1)