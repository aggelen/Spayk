#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:00:25 2022

@author: aggelen
"""
from Spayk.Core import Simulator
from Spayk.Stimuli import ConstantCurrentSource
from Spayk.Organization import Tissue
from Spayk.Nerves import SingleIzhikevichNeuron
# from Spayk.Observation import ObservationSettings

#%% Create Network
tissue = Tissue()

# n0 = SingleIzhikevichNeuron(stimuli=ConstantCurrentSource(20))
# n1 = SingleIzhikevichNeuron(stimuli=ConstantCurrentSource(10))
# tissue.add([n0,n1])

# random izh. neurons with multiple dynamics
import numpy as np

neuron_dynamics = {0: 'regular_spiking',
                   1: 'intrinsically_bursting',
                   2: 'chattering',
                   3: 'fast_spiking',
                   4: 'thalamo_cortical',
                   5: 'resonator',
                   6: 'low_threshold_spiking'}

random_neurons = []
for i in range(100):
    random_neurons.append(SingleIzhikevichNeuron(stimuli=ConstantCurrentSource(10),
                                                 dynamics=neuron_dynamics[np.random.randint(7)]))
tissue.add(random_neurons)

#%% Embody
tissue.embody()

#%% Observation Settings
settings = {'duration': 100,
            'dt': 0.1}

#%% Run
sim0 = Simulator()
sim0.keep_alive(tissue, settings)

#%% Aux.
# tissue.neurons[0].plot_v(dt=0.1)
# tissue.neurons[1].plot_v(dt=0.1)

tissue.plot_membrane_potential_of(0, dt=0.1, color='g')
tissue.raster_plot(dt=0.1)

