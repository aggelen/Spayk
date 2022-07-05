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

# An experiment for re-produce results of
# Spike timing dependent plasticity finds the start of repeating patterns in continuous spike trains.
# Timothée Masquelier, Rudy Guyonneau, Simon J Thorpe

#%% Stimuli
# varying spike rates between 0 - 90hz, change speed +-360 Hz/s, cliped between: +- 1800 Hz/s
no_neurons = 2000
dt = 1.0   #ms->s
t_stop = 15000 #ms

r = np.random.uniform(0, 90, size=(no_neurons))
s = np.random.uniform(-1800, 1800, size=(no_neurons))
spike_train = []
for t in range(t_stop):
    prob = np.random.uniform(0, 1, r.shape)
    spikes = np.less(prob, np.array(r)*dt*1e-3)
    spike_train.append(spikes)
    r = np.clip(r + s*dt*1e-3 , 0, 90)
    ds = np.random.uniform(-360, 360, size=(no_neurons))
    s = np.clip(s + ds, -1800, 1800)

spike_train = np.array(spike_train).T

#%% 50ms check

# t = np.arange(0,t_stop+50,50)
# for i in range(t.size - 1):
#     chunk50ms = spike_train[:, t[i]:t[i+1]]
#     no_spike_idx = np.sum(chunk50ms, 1) == 0
    
#     template = np.zeros_like(chunk50ms[no_spike_idx]).astype(bool)
#     template[:,25] = True
    
#     chunk50ms[no_spike_idx] = template

for t in range(t_stop):
    if t > 50:
        no_firing_idx = np.count_nonzero(spike_train[:,t-50:t],1) == 0
        if no_firing_idx.sum():
            spike_train[no_firing_idx,t] = 1

#jitter
jitter_prob = np.random.uniform(0,1,spike_train.shape)
jitter = jitter_prob < 10*dt*1e-3

spike_train = np.logical_or(spike_train, jitter)
    
#%% repeating
repeating_pattern = spike_train[:1000, 25:75]

repeat_times = [25]
last_repeat_time = 25
for t in range(t_stop-50):
    if t > last_repeat_time + np.random.uniform(75,225,1):
        not_placed = True
        while(not_placed):
            if np.random.uniform(0,1) < 0.25:
                repeat_times.append(t)
                last_repeat_time = t
                not_placed = False

                spike_train[:1000, t:t+50] = np.copy(repeating_pattern)

repeat_times = np.array(repeat_times)



stimuli = ExternalSpikeTrain(dt, t_stop, no_neurons, spike_train)

#%% Custom Raster Plot
fig, ax = plt.subplots()
spike_loc = np.argwhere(spike_train[:,:500])
ax.scatter(spike_loc[:,1]*dt, spike_loc[:,0], s=2.5)

from matplotlib.patches import Rectangle

start = repeat_times[repeat_times<450] + 25
end = np.ones_like(start)*500
a = tuple(np.vstack((start,end)))
width = 50
height = 1000
for a_x, a_y in zip(*a):
    ax.add_patch(Rectangle(xy=(a_x-width/2, a_y-height/2) ,width=width, height=height, linewidth=1, color='gray', fill=True, alpha=0.4))

#%% Neuron
stdp_params = {'a_plus': 0.03125, 'a_minus': 0.029, 'tau_plus': 16.8, 'tau_minus': 33.7}
n_params = {'n_synapses': no_neurons,
            'dt': 1.0,
            'w': np.full((no_neurons), 0.475, dtype=np.float32),
            'stdp_on': True,
            'stdp_params': stdp_params}

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
end = np.ones_like(start)*500
a = tuple(np.vstack((start,end)))
width = 50
height = 1000
for a_x, a_y in zip(*a):
    ax.add_patch(Rectangle(xy=(a_x-width/2, a_y-height/2) ,width=width, height=height, linewidth=1, color='gray', fill=True, alpha=0.4))
