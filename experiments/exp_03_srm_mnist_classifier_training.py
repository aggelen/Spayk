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
import seaborn as sns
sns.set()
plt.close('all')

# Classification experiment with 3 srm lif neurons

#%% Stimuli: randomly placed 5 patterns under presence of other signals! 
## highly important!
# dataset contains random (t_stop/t_sample) samples of 3 classes
dt = 1.0
n_samples = 50
no_neurons = 784*2
t_stop = 50000
dataset = SpikingMNIST()
spike_train, repeat_times, repeat_mode = dataset.special_spike_train(dt, t_stop=t_stop, t_sample=50)

#%%
stimuli = ExternalSpikeTrain(dt, t_stop, 784*2, spike_train)
# stimuli.raster_plot()

#%% Custom Raster Plot
fig, ax = plt.subplots()
spike_loc = np.argwhere(spike_train)
ax.scatter(spike_loc[:,1]*dt, spike_loc[:,0], s=2.5)

from matplotlib.patches import Rectangle

start = repeat_times + 25
end = np.ones_like(start)*784/2
a = tuple(np.vstack((start,end)))
width = 50
height = 784
for a_x, a_y in zip(*a):
    ax.add_patch(Rectangle(xy=(a_x-width/2, a_y-height/2) ,width=width, height=height, linewidth=1, color='gray', fill=True, alpha=0.4))
    
#%% Neuron
stdp_params = {'a_plus': 0.03125, 'a_minus': 0.028, 'tau_plus': 16.8, 'tau_minus': 33.7}
n_params = {'n_synapses': no_neurons,
            'dt': 1.0,
            'w': np.full((no_neurons), 0.475, dtype=np.float32),
            'stdp_on': True,
            'stdp_params': stdp_params,
            'v_th': 135}

recog_neuron = SRMLIFNeuron(n_params)

# bind neuron to a tissue
recognation_tissue = Tissue([recog_neuron])

# run simulation
recognation_tissue.keep_alive(stimuli=stimuli)
recognation_tissue.logger.plot_v()

#%%
from matplotlib.patches import Rectangle
ax = plt.gca()
start = repeat_times + 25
end = np.ones_like(start)*100
a = tuple(np.vstack((start,end)))
width = 50
height = 350
for a_x, a_y in zip(*a):
    ax.add_patch(Rectangle(xy=(a_x-width/2, a_y-height/2) ,width=width, height=height, linewidth=1, color='gray', fill=True, alpha=0.4))
    
#%%
plt.figure()
plt.plot(recog_neuron.w_mean)
plt.grid()

