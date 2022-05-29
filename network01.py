#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:00:25 2022

@author: aggelen
"""
from spayk.Core import Simulator
from spayk.Stimuli import ExternalCurrentSignal
from spayk.Organization import Tissue
from spayk.Nerves import SingleIzhikevichNeuron
from spayk.Synapses import Synapse, GENESIS_Synapse
# from Spayk.Observation import ObservationSettings

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#%% Stimuli
triangle = np.array([[0,0,0,0,0],
                     [0,1,0,0,0],
                     [0,1,1,0,0],
                     [0,1,1,1,0],
                     [0,0,0,0,0]])

frame = np.array([[1,1,1,1,1],
                  [1,0,0,0,1],
                  [1,0,0,0,1],
                  [1,0,0,0,1],
                  [1,1,1,1,1]])

square = np.array([[0,0,0,0,0],
                   [0,1,1,1,0],
                   [0,1,1,1,0],
                   [0,1,1,1,0],
                   [0,0,0,0,0]])

noise_sigma = 0
tri_input_currents = triangle.flatten()*20 + noise_sigma*np.random.randn(len(triangle.flatten()))
frm_input_currents = frame.flatten()*20 + noise_sigma*np.random.randn(len(frame.flatten()))
sqr_input_currents = square.flatten()*20 + noise_sigma*np.random.randn(len(square.flatten()))

# total sim 1000 ms, dt = 0.1, 500ms triangle, 500 ms square
signal_tri = np.tile(tri_input_currents.reshape(-1,1), 5000)
signal_frm = np.tile(frm_input_currents.reshape(-1,1), 5000)
signal_sqr = np.tile(sqr_input_currents.reshape(-1,1), 5000)
signal = np.hstack((signal_tri, signal_frm, signal_sqr))

#%% tissue for input
eye = Tissue()
eye_neurons = []
for i in range(25):
    eye_neurons.append(SingleIzhikevichNeuron(stimuli=ExternalCurrentSignal(signal[i])))
    
eye.add(eye_neurons)
eye.embody()

#%% connections 
# 100 neurons randomly connected to each other
connections = {}
synapses = []
neuron_idx = np.arange(100)
exc_inh_statuses = np.append(np.ones(60), -np.ones(40))
for i in range(100):
    available = np.delete(neuron_idx, i)
    random_inputs = np.random.choice(np.arange(25), 10)
    random_conns = np.random.choice(available, 50)
    connections[i] = [random_conns, random_inputs]
    synapses.append(GENESIS_Synapse(params={'tauExc': 15, 'tauInh': 10, 'dt': 0.1},
                                    io_neuron_idx=[random_conns, i],
                                    external_inputs=random_inputs,
                                    exc_inh=exc_inh_statuses[random_conns]))

#%%tissue for network
# gs_01 = GENESIS_Synapse(params={'tauExc': 10, 'dt': 0.1}, io_neuron_idx=[input_neuron_idx, output_neuron_idx])
# n_postsyn = SingleIzhikevichNeuron(stimuli=gs_01)

network0 = Tissue(connected=[eye])
neurons = []
for i in range(100):
    if i<80:
        neurons.append(SingleIzhikevichNeuron(stimuli=synapses[i]))
    else:
        neurons.append(SingleIzhikevichNeuron(stimuli=synapses[i],
                                              dynamics='chattering'))
network0.add(neurons)
network0.embody()

#%%tissue for output
output_tissue = Tissue(connected=[network0])

op_synapses = []
op_synapses.append(GENESIS_Synapse(params={'tauExc': 15, 'tauInh': 10, 'dt': 0.1},
                        io_neuron_idx=[np.append(np.ones(25), 2*np.ones(25)).astype(int), 0],
                        external_inputs=np.random.randint(0,100,50),
                        exc_inh=-np.ones(50)))
op_synapses.append(GENESIS_Synapse(params={'tauExc': 15, 'tauInh': 10, 'dt': 0.1},
                        io_neuron_idx=[0*np.append(np.ones(25), 2*np.ones(25)).astype(int), 1],
                        external_inputs=np.random.randint(0,100,50),
                        exc_inh=-np.ones(50)))
op_synapses.append(GENESIS_Synapse(params={'tauExc': 15, 'tauInh': 10, 'dt': 0.1},
                        io_neuron_idx=[np.append(0*np.ones(25), 1*np.ones(25)).astype(int), 2],
                        external_inputs=np.random.randint(0,100,50),
                        exc_inh=-np.ones(50)))

for i in range(3):
    op_synapses[i].set_W(np.ones(50))

neurons = []
for i in range(3):
    neurons.append(SingleIzhikevichNeuron(stimuli=op_synapses[i]))
   
output_tissue.add(neurons)
output_tissue.embody()

#%% Observation Settings
settings = {'duration': 1500,
            'dt': 0.1}

#%% Run
sim0 = Simulator()
sim0.keep_alive([eye, network0, output_tissue], settings)

#%% Vis.
eye.raster_plot(dt=0.1)
network0.raster_plot(dt=0.1)
output_tissue.raster_plot(dt=0.1)
