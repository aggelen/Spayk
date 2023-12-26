#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:49:01 2023

@author: gelenag
"""

from spayk.Models import SRMLIFNeuron, IzhikevichNeuronGroup
from spayk.Stimuli import SpikeInstance
from spayk.Organization import Tissue
import numpy as np

stim_freq = 30 # Hz
neuron = SRMLIFNeuron({'n_synapses': 9, 'dt': 1/stim_freq})

group_params = {'no_neurons': 1,
                'behaviour': 'synaptic',
                'no_syn': 9,
                'E': np.zeros(9)}

neuron = IzhikevichNeuronGroup(group_params)
neuron.autoconnect()

stim = SpikeInstance(dt=1/stim_freq)
ca1 = Tissue([neuron])

op = []

for i in range(100):
    spikes = np.random.randint(0,2,9)
    stim.load_spikes(spikes)
    ca1.keep_alive(stim)

    op.append(ca1.output)
    
op = np.array(op)[:,:,0]
import matplotlib.pyplot as plt
plt.plot(op)