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

#%% Model0
# create izhikevich neurons with different dynamics
neurongroup0 = IzhikevichNeuronGroup(1)

# AUTOCONNECT -> create synaptic connections and bind
neurongroup0.autoconnect()
auto_connected_tissue = Tissue([neurongroup0])

#%% Model1
neurongroup1 = IzhikevichNeuronGroup(1)

# Custimization
neurongroup1.set_architecture()
manually_connected_tissue = Tissue([neurongroup1])

#%% Model2
neurongroup2 = IzhikevichNeuronGroup(1)

# Custimization
neurongroup2.autoconnect()
recurrent_tissue = Tissue([neurongroup2])

#%% Simulations
# run simulation
sim_params = {'dt': 0.1}
auto_connected_tissue.keep_alive(sim_params)