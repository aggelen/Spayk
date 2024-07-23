#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:30:19 2024

@author: gelenag
"""
import numpy as np

class LIFGroup:
    def __init__(self, no_neurons, group_label, state_vector, params):
        self.group_label = group_label
        self.no_neurons = no_neurons
        self.state_vector = state_vector
        self.params = params
        
        # table with cols AMPA AMPAEXT NMDA GABA
        self.neuron_channel_assoc_table = np.zeros((self.no_neurons, 4))
        
        self.neuron_labels = [str(group_label)+str(i) for i in range(no_neurons)]
