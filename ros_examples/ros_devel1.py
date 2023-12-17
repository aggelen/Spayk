#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:49:01 2023

@author: gelenag
"""

from spayk.Models import SRMLIFNeuron, IzhikevichNeuronGroup
from spayk.Stimuli import SpikeInstance
from spayk.Organization import Tissue
from spayk.Learning import STDP
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

stdp_params = {'a_plus': 0.03125, 'a_minus': 0.025, 'tau_plus': 16.8, 'tau_minus': 30.7, 'dt': 1/stim_freq}
plasticity = STDP(stdp_params)

# new_weights = plasticity.offline(pre_spikes=input_spike_train.spikes, 
#                                  post_spikes=sup_sig.spikes, 
#                                  weights=neurongroup.w,
#                                  learning_rate=5e-3,
#                                  log_changes=True)

op = []
for i in range(100):
    spikes = np.random.randint(0,2,9)
    stim.load_spikes(spikes)
    ca1.keep_alive(stim)

    op.append(ca1.output)
    
op = np.array(op)[:,:,0]
import matplotlib.pyplot as plt
plt.plot(op)