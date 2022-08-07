#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:47:17 2022

@author: aggelen
"""

import sys
sys.path.append('..')

from spayk.Organization import Tissue
from spayk.Models import IzhikevichNeuronGroup
from spayk.Stimuli import SpikingMNIST, ExternalSpikeTrain
from spayk.Learning import STDP

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# stimuli
dataset = SpikingMNIST()
label_0, img0 = dataset.get_random_sample(label=1, dt=0.1, t_stop=500, jitter=2, max_fr=25)
label_1, img1 = dataset.get_random_sample(label=4, dt=0.1, t_stop=500, jitter=2, max_fr=25)
label_2, img2 = dataset.get_random_sample(label=7, dt=0.1, t_stop=500, jitter=2, max_fr=25)
st = np.hstack([label_0, label_1, label_2, label_1, label_0, label_2])
input_spike_train = ExternalSpikeTrain(dt=0.1, t_stop=6*500, no_neurons=st.shape[0], spike_train=st)
input_spike_train.raster_plot(title='Input Stimuli (MNIST Labels)', title_pad=60)

from matplotlib.patches import Rectangle
ax = plt.gca()
start = [250,750,1250,1750,2250,2750]
end = [125,125,125,125,125,125]
a = tuple(np.vstack((start,end)))
width = 490
height = 250
colors = ['gray','blue','red','blue','gray','red']
i=0
for a_x, a_y in zip(*a):
    ax.add_patch(Rectangle(xy=(a_x-width/2, a_y-height/2), width=width, height=height, linewidth=1, color=colors[i], fill=True, alpha=0.1))
    i+=1
    
plt.ylim([0,250])
plt.xlim([0,3000])
# plt.rcParams["figure.figsize"] = [3.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.gcf()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

ctr = (np.array([250,700,1170,1600,2075,2550])/3000)
vert = 0.85
newax = fig.add_axes([ctr[0],vert,0.1,0.1], anchor='C', zorder=1)
newax.imshow(-(img0-255), cmap='gray')
newax.axis('off')

newax = fig.add_axes([ctr[1],vert,0.1,0.1], anchor='C', zorder=1)
newax.imshow(-(img1-255), cmap='gray')
newax.axis('off')

newax = fig.add_axes([ctr[2],vert,0.1,0.1], anchor='C', zorder=1)
newax.imshow(-(img2-255), cmap='gray')
newax.axis('off')

newax = fig.add_axes([ctr[3],vert,0.1,0.1], anchor='C', zorder=1)
newax.imshow(-(img1-255), cmap='gray')
newax.axis('off')

newax = fig.add_axes([ctr[4],vert,0.1,0.1], anchor='C', zorder=1)
newax.imshow(-(img0-255), cmap='gray')
newax.axis('off')

newax = fig.add_axes([ctr[5],vert,0.1,0.1], anchor='C', zorder=1)
newax.imshow(-(img2-255), cmap='gray')
newax.axis('off')

#%% Synaptic Model
no_inh_neurons = 75
no_syn = input_spike_train.no_neurons
inh_synapse_idx = np.random.choice(no_syn, size=no_inh_neurons, replace=False)
E = np.zeros(no_syn)
E[inh_synapse_idx] = -75

group_params = {'no_neurons': 1,
                'behaviour': 'synaptic',
                'no_syn': no_syn,
                'E': E}

neurongroup = IzhikevichNeuronGroup(group_params)
neurongroup.autoconnect(scale=2.0)

tissue = Tissue([neurongroup])

tissue.keep_alive(stimuli=input_spike_train)
tissue.logger.plot_v()
output_spikes = tissue.logger.output_spikes()

#%% Offline STDP Trainer
stdp_params = {'a_plus': 0.03125, 'a_minus': 0.029, 'tau_plus': 16.8, 'tau_minus': 33.7, 'dt': 0.1}
plasticity = STDP(stdp_params)
new_weights = plasticity.offline(pre_spikes=input_spike_train.spikes, 
                                 post_spikes=output_spikes, 
                                 weights=neurongroup.w)


#%% Repeat experiment
input_spike_train.reset()
neurongroup.w = new_weights
tissue = Tissue([neurongroup])

tissue.keep_alive(stimuli=input_spike_train)
tissue.logger.plot_v()
