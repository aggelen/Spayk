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

#%% Stimuli: randomly placed 5 patterns under presence of other signals! 
## highly important!
# dataset contains random (t_stop/t_sample) samples of 3 classes
dt = 1.0
n_samples = 50
no_neurons = 784*2
dataset = SpikingMNIST()
spike_train, repeat_times, repeat_labels = dataset.special_spike_train(dt, 
                                                                       t_stop=15000, 
                                                                       t_sample=50, 
                                                                       distribute_5=True)

#%%
stimuli = ExternalSpikeTrain(dt, 15000, 784*2, spike_train)
# stimuli.raster_plot()

#%% Custom Raster Plot
fig, ax = plt.subplots()
spike_loc = np.argwhere(spike_train)
ax.scatter(spike_loc[:,1]*dt, spike_loc[:,0], s=2.5)

from matplotlib.patches import Rectangle

start = repeat_times + 25
end = np.ones_like(start)*784/2
a = tuple(np.vstack((start.flatten(),end.flatten())))
width = 50
height = 784
for a_x, a_y in zip(*a):
    ax.add_patch(Rectangle(xy=(a_x-width/2, a_y-height/2) ,width=width, height=height, linewidth=1, color='gray', fill=True, alpha=0.4))
    
#%% Neuron
w = np.load('w_5.npy', allow_pickle=True)
n_params = {'n_synapses': no_neurons,
            'dt': 1.0,
            'w': w,
            'v_th': 135}

recog_neuron = SRMLIFNeuron(n_params)

# bind neuron to a tissue
recognation_tissue = Tissue([recog_neuron])

# run simulation
recognation_tissue.keep_alive(stimuli=stimuli)
recognation_tissue.logger.plot_v()

#%%
repeat_times = np.argwhere(repeat_labels == 5)*50

from matplotlib.patches import Rectangle
ax = plt.gca()
start = repeat_times + 25
end = np.ones_like(start)*100
a = tuple(np.vstack((start.flatten(),end.flatten())))
width = 50
height = 350
for a_x, a_y in zip(*a):
    ax.add_patch(Rectangle(xy=(a_x-width/2, a_y-height/2) ,width=width, height=height, linewidth=1, color='gray', fill=True, alpha=0.4))
    
#%%
plt.figure()
plt.plot(recog_neuron.w_mean)

