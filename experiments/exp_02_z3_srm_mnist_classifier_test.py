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
from spayk.Stimuli import SpikingMNIST, ExternalSpikeTrain

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Classification experiment with 3 srm lif neurons

#%% Stimuli: 3 classes as triangle, square and frame encoded in 25 synaptic connection
# dataset contains random (t_stop/t_sample) samples of 3 classes
dt = 1.0
n_samples = 50
dataset = SpikingMNIST()
test_train, test_labels = dataset.generate_test_spike_train(n_samples=n_samples, dt=dt, t_sample=50)
no_neurons = 784*2
t_stop = test_train.shape[1]

r = np.random.uniform(0, 30, size=(no_neurons))
print('first r: {}'.format(r.mean()))
s = np.random.uniform(-50, 50, size=(no_neurons))
print('first s: {}'.format(s.mean()))
spike_train = []
for t in range(t_stop):
    prob = np.random.uniform(0, 1, r.shape)
    spikes = np.less(prob, np.array(r)*dt*1e-3)
    spike_train.append(spikes)
    r = np.clip(r + s*dt*1e-3 , 0, 30)
    ds = np.random.uniform(-5, 5, size=(no_neurons))
    s = np.clip(s + ds, -50, 50)

spike_train = np.array(spike_train).T
spike_train[:784,:] = test_train
test_stimuli = ExternalSpikeTrain(dt, t_stop, no_neurons, spike_train)
test_stimuli.raster_plot()

#%%
ws = np.load('ws.npy', allow_pickle=True)

n_params = {'n_synapses': no_neurons,
            'dt': 1.0,
            'w': ws[2],
            'v_th': 135}

recog_neuron = SRMLIFNeuron(n_params)
recognation_tissue = Tissue([recog_neuron])
recognation_tissue.keep_alive(test_stimuli)
recognation_tissue.logger.plot_v()