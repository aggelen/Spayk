#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:47:17 2022

@author: aggelen
"""

import sys
sys.path.append('..')

from spayk.Organization import Tissue
from spayk.Models import SRMLIFNeuronGroup
from spayk.Stimuli import SpikingClassificationDataset

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Classification experiment with 3 srm lif neurons

#%% Stimuli: 3 classes as triangle, square and frame encoded in 25 synaptic connection
# dataset contains random (t_stop/t_sample) samples of 3 classes
dataset = SpikingClassificationDataset(dt=1.0, t_sample=50, t_stop=30000)
dataset.spike_train.raster_plot()
stimuli = dataset.spike_train

#%% Neuron
ap = 0.03225
stdp_params = {'a_plus': ap, 'a_minus': 0.85*ap, 'tau_plus': 16.8, 'tau_minus': 33.7}
super_stim = dataset.target_spike_train.spikes
n1_params = {'n_synapses': stimuli.no_neurons,
             'dt': 1.0,
             'w': np.full((stimuli.no_neurons), 0.475, dtype=np.float32),
             'stdp_on': True,
             'stdp_params': stdp_params,
             'supervise_on': True}

n2_params = n1_params.copy()
n3_params = n1_params.copy()
n1_params['supervision_stimuli'] = super_stim[0]
n2_params['supervision_stimuli'] = super_stim[1]
n3_params['supervision_stimuli'] = super_stim[2]
neuron_params = [n1_params, n2_params, n3_params]

group_params = {'no_neurons': 3}
classification_neurons = SRMLIFNeuronGroup(group_params, neuron_params)

# bind neuron to a tissue
classification_tissue = Tissue([classification_neurons])

# run simulation
classification_tissue.keep_alive(stimuli=stimuli)
classification_tissue.logger.plot_v()

#%% Test
test_dataset = SpikingClassificationDataset(dt=1.0, t_sample=50, t_stop=150)
test_stimuli = test_dataset.spike_train

classification_tissue.neuron_group.neurons[0].count = 0
classification_tissue.neuron_group.neurons[1].count = 0
classification_tissue.neuron_group.neurons[2].count = 0

classification_tissue.keep_alive(stimuli=test_stimuli)
classification_tissue.logger.plot_v()