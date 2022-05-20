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

# tissue.plot_membrane_potential_of(0, dt=0.1, color='g')
tissue.raster_plot(dt=0.1)

#%% synapse devel
syn0 = Synapse()
AMPA_decay_params = {'g_syn0': 5,
                     'tau': 2,
                     'tf': 1000}

t = np.arange(-10,500, 0.1)
time_array = np.arange(0,100,0.1)

# syn0.create_channel(AMPA_decay_params, descriptor='AMPA', model='single_decay')

# spike_times = np.where(tissue.log_spikes[0])[0]


# Isyn = []
# for tid, t in enumerate(time_array):
#     if tid in spike_times:
#         syn0.channels['AMPA'].tf = t
        
#     ie = -syn0.calculate_syn_current(t, syn0.channels['AMPA'], u=-65, Esyn=0)
#     Isyn.append(ie)
    
# plt.figure()
# plt.plot(time_array, Isyn)

#%% Genesis
gs0 = GENESIS_Synapse(tau=10, dt=0.1)

gSyn = []
for tid, t in enumerate(time_array):
    gSyn.append(gs0.g_syn(spike=tissue.log_spikes[0][tid],t=t))

plt.figure()
plt.plot(time_array, gSyn)