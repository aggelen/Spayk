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


#%% Stimuli
dataset = SpikingMNIST()

def mnist_spike_train(dataset, label, dt, t_stop):
    no_neurons = 784*2
    r = np.random.uniform(0, 30, size=(no_neurons))
    s = np.random.uniform(-50, 50, size=(no_neurons))
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
    pattern = dataset.get_random_sample(label, dt=dt, t_stop=50)

    #%% repeating
    first_repeat = 25
    spike_train[:784, first_repeat:first_repeat+50] = np.copy(pattern)

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
            spike_train[:784, last_repeat_time:last_repeat_time+50] = np.copy(pattern)

    repeat_times = np.array(repeat_times)
    stimuli = ExternalSpikeTrain(dt, t_stop, no_neurons, spike_train)
    
    return stimuli, repeat_times

#%% for every label (0-9), find weights via stdp
ws = []
for label in range(10):
    stimuli, repeat_times = mnist_spike_train(dataset, label=label, dt=1.0, t_stop=15000)
    
    #%% Neuron
    stdp_params = {'a_plus': 0.03125, 'a_minus': 0.028, 'tau_plus': 16.8, 'tau_minus': 33.7}
    n_params = {'n_synapses': stimuli.no_neurons,
                'dt': 1.0,
                'w': np.full((stimuli.no_neurons), 0.475, dtype=np.float32),
                'stdp_on': True,
                'stdp_params': stdp_params,
                'v_th': 135}

    recog_neuron = SRMLIFNeuron(n_params)

    # bind neuron to a tissue
    recognation_tissue = Tissue([recog_neuron])

    # run simulation
    recognation_tissue.keep_alive(stimuli=stimuli)
    recognation_tissue.logger.plot_v()
    
    ws.append(recog_neuron.w)

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


ws = np.array(ws)

# çalışanlar
# 0,2,3,7,8,9