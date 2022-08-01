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
from spayk.Stimuli import ExternalSpikeTrain, SpikingMNIST, poisson_spike_train_generator

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#%% Stimuli
# varying spike rates between 0 - 90hz, change speed +-360 Hz/s, cliped between: +- 1800 Hz/s
no_neurons = 784*2
dt = 1.0   #ms->s
t_stop = 15000 #ms

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

#%% FIXME! Unsable with random 5 patterns
dataset = SpikingMNIST()
five_pattern = dataset.get_random_sample(label=5, dt=dt, t_stop=50)

#%% repeating
first_repeat = 25
spike_train[:784, first_repeat:first_repeat+50] = np.copy(five_pattern)

repeat_times = [first_repeat]
last_repeat_time = first_repeat
stab_counter = 0
stab_idx = [120, 280, 510]
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
        spike_train[:784, last_repeat_time:last_repeat_time+50] = np.copy(five_pattern)

repeat_times = np.array(repeat_times)
stimuli = ExternalSpikeTrain(dt, t_stop, no_neurons, spike_train)
# stimuli.raster_plot()

#%% Custom Raster Plot
fig, ax = plt.subplots()
spike_loc = np.argwhere(spike_train[:,:500])
ax.scatter(spike_loc[:,1]*dt, spike_loc[:,0], s=2.5)

from matplotlib.patches import Rectangle

start = repeat_times[repeat_times<450] + 25
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

#%%
input_frates = np.sum(spike_train,1)/(t_stop/1000)
plt.figure()
plt.plot(input_frates)

plt.figure()
# plt.hist(input_frates,40)
plt.hist(np.sum(five_pattern, 1),10)
