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
from spayk.Stimuli import PoissonSpikeTrain

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# stimuli
# input_spike_train = PoissonSpikeTrain(dt=0.1, t_stop=1000, no_neurons=100, spike_rates=np.random.randint(5,20,100))
input_spike_train = PoissonSpikeTrain(dt=0.1, t_stop=1000, no_neurons=100, spike_rates=2*np.ones(100))
input_spike_train.raster_plot()

#%% Model0: single izhikevich neuron have 100 syn connections
# different behaviours, simple, synaptic, stdp

# group_params = {'no_neurons': 1,
#                 'behaviour': 'synaptic',
#                 'no_syn': 100}

# synaptic_neuron = IzhikevichNeuronGroup(group_params)

# # AUTOCONNECT -> create synaptic connections and bind
# synaptic_neuron.autoconnect()
# synaptic_neuron_tissue = Tissue([synaptic_neuron])

# synaptic_neuron_tissue.keep_alive(stimuli=input_spike_train)
# synaptic_neuron_tissue.logger.plot_v()


#%% Model1: many izhikevich neurons as a layer, many synaptic inputs
no_neurons = 1000
group_params = {'no_neurons': no_neurons,
                'behaviour': 'synaptic',
                'no_syn': 100}
neurongroup1 = IzhikevichNeuronGroup(group_params)

# Custimization
ihn_neuron_idx = np.random.choice(np.arange(no_neurons), 200)   # 200 inhibitory
dynamics = np.zeros(no_neurons)
dynamics[ihn_neuron_idx] = 3    # inh neurons are chattering!

w = np.zeros((1000,100), dtype=np.float32)
w[ np.random.uniform(0,1,(1000,100)) < 0.1 ] = 0.07

custumization_params = {'dynamics': dynamics, 'w': w}
neurongroup1.set_architecture(custumization_params)
customized_tissue = Tissue([neurongroup1])

customized_tissue.keep_alive(stimuli=input_spike_train)

#colorize inh.
neuron_types = np.zeros(1000)
neuron_types[ihn_neuron_idx] = 1
customized_tissue.logger.raster_plot(color_array=neuron_types)

#%% Model2: recurrent tissue
input_spike_train.reset()
no_neurons = 1000
group_params = {'no_neurons': no_neurons,
                'behaviour': 'recurrent',
                'no_syn': 100}
rec_neurongroup = IzhikevichNeuronGroup(group_params)

# Custimization
# experiment inspired from kaizouman trick! (https://github.com/kaizouman/tensorsandbox)
ihn_neuron_prob = np.random.uniform(0,1,1000)
ihn_neuron_idx = np.argwhere(ihn_neuron_prob < 0.2)
# ihn_neuron_idx = np.random.choice(np.arange(no_neurons), 200)   # 200 inhibitory
dynamics = np.zeros(no_neurons)
dynamics[ihn_neuron_idx] = 3    # inh neurons are chattering!

w = np.zeros((1000,100), dtype=np.float32)
w[ np.random.uniform(0,1,(1000,100)) < 0.1 ] = 0.07

# recurrent arch.
w_rec = np.zeros((1000,1000),  dtype=np.float32)
rec_prob = np.random.uniform(0,1,(1000,1000))
w_rec[rec_prob < 0.1] = np.random.gamma(2, 0.003, size=w_rec[rec_prob < 0.1].shape)

# Inhibitory to excitatory connections are twice as strong. (kaizouman)
inh_2_exc = np.ix_(ihn_neuron_prob >= 0.2, ihn_neuron_prob < 0.2)
w_rec[ inh_2_exc ] = 2*w_rec[ inh_2_exc]

E_rec = np.zeros((1000), dtype=np.float32)
E_rec[ihn_neuron_idx] = -85.0

custumization_params = {'dynamics': dynamics, 'w': w, 'E_rec': E_rec, 'w_rec': w_rec}
rec_neurongroup.set_architecture(custumization_params)
rec_tissue = Tissue([rec_neurongroup])

rec_tissue.keep_alive(stimuli=input_spike_train)

#colorize inh.
neuron_types = np.zeros(1000)
neuron_types[ihn_neuron_idx] = 1
rec_tissue.logger.raster_plot(neuron_types)

#%%
import matplotlib.lines as mlines
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
eight = mlines.Line2D([], [], color=colors[0], marker='.', ls='', label='Exc.')
nine = mlines.Line2D([], [], color=colors[1], marker='.', ls='', label='Inh.')
# etc etc
plt.legend(handles=[eight, nine], loc='upper right')