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

n0 = SingleIzhikevichNeuron(stimuli=ConstantCurrentSource(20))
tissue.add([n0])

#%% Embody
tissue.embody()

#%% Observation Settings
settings = {'duration': 100,
            'dt': 0.1}

#%% Run
sim0 = Simulator()
sim0.keep_alive(tissue, settings)

#%% Aux.
tissue.neurons[0].plot_v(dt=0.1)