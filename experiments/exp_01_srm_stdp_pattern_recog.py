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
from spayk.Stimuli import ExternalSpikeTrain, Masquelier2008SpikeTrain

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
for t in range(t_stop):
    no_firing_idx = np.count_nonzero(spike_train[:,t-50:t],1) == 0
    if t >= 50:
        if no_firing_idx.sum():
            spike_train[no_firing_idx,t] = True
    
#%% repeating
first_repeat = 65
repeating_pattern = spike_train[:1000, first_repeat:first_repeat+50]

repeat_times = [first_repeat]
last_repeat_time = first_repeat
for t in range(t_stop-50):
    if t > last_repeat_time + 100:
        r = np.random.uniform(0,1,25)
        mask = r < 0.25
        c = np.argwhere(mask).min() 
        start_at = t + c*50
 
        if start_at + 50 > t_stop:
            break
        
        repeat_times.append(start_at) 
        last_repeat_time = start_at
        spike_train[:1000, last_repeat_time:last_repeat_time+50] = np.copy(repeating_pattern)

repeat_times = np.array(repeat_times)

# + 10 Hz jitter
jitter_prob = np.random.uniform(0,1,spike_train.shape)
jitter = jitter_prob < 10*dt*1e-3

spike_train = np.logical_or(spike_train, jitter)

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
