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

neurons = []
for i in range(4):
    neurons.append(SingleIzhikevichNeuron(stimuli=ConstantCurrentSource(np.random.randint(8,16)),
                                          dynamics=neuron_dynamics[0]))
tissue.add(neurons)
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

#%% Synapse development
"""
Isyn_j = sum{i=for all presyn neurons}{Iij * wij}

4 neurons - 3 synapses [w13, w24, w34]

all excitatory: ie = i_NMDA + i_AMPA
"""
W = np.zeros((4,4))

synaptic_conns = np.array([[0,0,0,0],
                           [0,0,0,0],
                           [1,0,0,0],
                           [0,1,1,0]])

W[synaptic_conns.astype(bool)] = np.random.rand(3)

ies = np.ones(4)
gsyns = np.ones_like(tissue.neurons[0].v_history)

#%% Synapse Devel.
import matplotlib.pyplot as plt
from spayk.Synapses import Synapse
plt.close('all')
syn0 = Synapse()

GABAa_single_decay_params = {'g_syn0': 40,
                             'tau': 5,
                             'tf': 0}
syn0.create_channel(GABAa_single_decay_params, descriptor='GABAa', model='single_decay')

GABAa_double_decay_params = {'g_syn0': 40,
                             'tau': 5,
                             'tf': 0,
                             'tau_fast': 6,
                             'tau_slow': 100,
                             'tau_rise': 1,
                             'a': 1}
syn0.create_channel(GABAa_double_decay_params, descriptor='GABAa2', model='double_decay')

t = np.arange(-10,500, 0.1)
y = -syn0.calculate_syn_current(t, syn0.channels['GABAa'], u=-65, Esyn=-70)
Isyn_GABA_a = -syn0.calculate_syn_current(t, syn0.channels['GABAa2'], u=-65, Esyn=-70)

plt.figure('Single vs Double Decay Model for GABA_A')
plt.plot(t[:900], y[:900])
plt.plot(t[:900], Isyn_GABA_a[:900])
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic Current (pA)')
plt.title('Single vs Double Decay Model for GABA_A, Spike @ t=0')
plt.legend(['Single', 'Double'])
plt.grid()

#%%
GABAb_double_decay_params = {'g_syn0': 40,
                             'tau': 5,
                             'tf': 0,
                             'tau_fast': 120,      # 100-300
                             'tau_slow': 500,      # 500-1000
                             'tau_rise': 50,       # 25-50
                             'a': 0.8}
syn0.create_channel(GABAb_double_decay_params, descriptor='GABAb', model='double_decay')
Isyn_GABA_b = -syn0.calculate_syn_current(t, syn0.channels['GABAb'], u=-65, Esyn=-70)
plt.figure()
plt.plot(t, Isyn_GABA_a)
plt.plot(t, Isyn_GABA_b)

#%%
AMPA_decay_params = {'g_syn0': 5,
                     'tau': 2,
                     'tf': 0}

syn0.create_channel(AMPA_decay_params, descriptor='AMPA', model='single_decay')
Isyn_AMPA = -syn0.calculate_syn_current(t, syn0.channels['AMPA'], u=-65, Esyn=0)
plt.plot(t, Isyn_AMPA)

#%%
NDMA_double_decay_params = {'g_syn0': 4,
                            'tau': 5,
                            'tf': 0,
                            'tau_fast': 40,     
                            'tau_slow': 200,      
                            'tau_rise': 5,       # 3-15
                            'a': 0.8}

syn0.create_channel(NDMA_double_decay_params, descriptor='NMDA', model='double_decay')
Isyn_NMDA = -syn0.calculate_syn_current(t, syn0.channels['NMDA'], u=-65, Esyn=0)
plt.plot(t, Isyn_NMDA)

plt.xlabel('Time (ms)')
plt.ylabel('Synaptic Current (pA)')
plt.title('Channel Models for AMPA, NMDA, GABA_A, GABA_B, Spike @ t=0')
plt.legend(['GABA_A', 'GABA_B', 'AMPA', 'NMDA'])
plt.grid()