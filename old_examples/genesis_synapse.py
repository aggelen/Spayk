#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:00:25 2022

@author: aggelen
"""
import sys
sys.path.append("..") 

from spayk.Core import Simulator
from spayk.Stimuli import ConstantCurrentSource
from spayk.Organization import Tissue
from spayk.Nerves import SingleIzhikevichNeuron
from spayk.Synapses import Synapse, GENESIS_Synapse
# from Spayk.Observation import ObservationSettings

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
#%% Create Network
tissue = Tissue()

n0 = SingleIzhikevichNeuron(stimuli=ConstantCurrentSource(np.random.randint(8,25)))
tissue.add([n0])

gs0 = GENESIS_Synapse(params={'tauExc': 15, 'dt': 0.1}, 
                        io_neuron_idx=[0, 1])
n_postsyn = SingleIzhikevichNeuron(stimuli=gs0)
tissue.add([n_postsyn])

#%% Embody
tissue.embody()

#%% Observation Settings
settings = {'duration': 100,
            'dt': 0.1}

#%% Run
sim0 = Simulator()
sim0.keep_alive(tissue, settings)

#%% Aux.
tissue.raster_plot(dt=0.1)

gs0.plot_channel_conductances(0, dt=0.1)