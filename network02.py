#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 00:08:30 2022

@author: aggelen
"""
from spayk.Core import Simulator
from spayk.Stimuli import ExternalCurrentSignal, SpikeTrains
from spayk.Organization import Tissue
from spayk.Nerves import SynapticIzhikevichNeuronGroup, RecurrentIzhikevichNeuronGroup
from spayk.Synapses import Synapse, GENESIS_Synapse
# from Spayk.Observation import ObservationSettings

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#%% Stimuli Generator
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

T = 500
dt = 0.1

spike_trains = SpikeTrains(25, r=triangle.flatten()*25, s_max=10)
stimuli_triangle = spike_trains.add_spikes(int(T/dt))
spike_trains = SpikeTrains(25, r=frame.flatten()*25, s_max=10)
stimuli_frame = spike_trains.add_spikes(int(T/dt))
spike_trains = SpikeTrains(25, r=square.flatten()*25, s_max=10)
stimuli_square = spike_trains.add_spikes(int(T/dt))
stimuli = np.vstack([stimuli_triangle, stimuli_frame, stimuli_square])
sc_spikes = np.argwhere(stimuli)
steps, neurons = sc_spikes.T
plt.scatter(steps*0.1, neurons, s=3)

#%% Network, 1000 neurons, %20: inhibitory
n = 1000
m = 25

p_neurons = np.random.uniform(0,1,(n))

a = np.full((n), 0.02, dtype=np.float32)
a[p_neurons < 0.2] = 0.1
d = np.full((n), 8.0, dtype=np.float32)
d[p_neurons < 0.2] = 2.0

# connect 25% of the neurons to the input synapses
p_syn = np.random.uniform(0,1,(n,m))
w_in = np.zeros((n,m), dtype=np.float32)

# w_in[ p_syn < 0.25 ] = 0.7
# w_in[ p_syn < 0.25 ] = np.random.randint(1,6, w_in[ p_syn < 0.25 ].shape)/10.0

w_in[ p_syn < 0.25 ] = 2*np.random.rand(w_in[ p_syn < 0.25 ].shape[0])-1

# Randomly distribute recurrent connections
w = np.zeros((n,n),  dtype=np.float32)
p_reccur = np.random.uniform(0,1,(n,n))
w[p_reccur < 0.1] = np.random.gamma(2, 0.03, size=w[p_reccur < 0.1].shape)
# Identify inhibitory to excitatory connections (receiving end is in row)
inh_2_exc = np.ix_(p_neurons >= 0.2, p_neurons < 0.2)
# Increase the strength of these connections
w[ inh_2_exc ] = 2* w[ inh_2_exc]

# Only inhibitory neurons have E=-85 mv
e = np.zeros((n), dtype=np.float32)
e[p_neurons<0.2] = -85.0

params = {'no_neurons': 1000,
          'dynamics': 'regular_spiking',
          'no_connected_neurons': m,
          'tau': 10.0,
          'A': a,
          'D': d,
          'W_in': w_in,
          'W': w,
          'E': e,
          'dt': 0.1}

network = RecurrentIzhikevichNeuronGroup(params)
 
#%% Simulate
t_stop = 1500
sim_params = {'dt': 0.1,
              't_stop': t_stop,
              'frate': 0.002,
              'stimuli': stimuli}

sim0 = Simulator()
sim0.new_core_syn(network, sim_params)

#%% Visualize
v_out = sim0.results['v_out']
dt = 0.1
# Split between inhibitory and excitatory
inh_v_out = np.where(p_neurons < 0.2, v_out, 0)
exc_v_out = np.where(p_neurons >= 0.2, v_out, 0)
# Identify spikes
inh_spikes = np.argwhere(inh_v_out == 35.0)
exc_spikes = np.argwhere(exc_v_out == 35.0)
# Display spikes over time

plt.figure()
plt.axis([0, t_stop, 0, n])
plt.title('Inhibitory and excitatory spikes')
plt.ylabel('Neurons')
plt.xlabel('Time (msec)')
# Plot inhibitory spikes
steps, neurons = inh_spikes.T

plt.scatter(steps*dt, neurons, s=3)
# Plot excitatory spikes
steps, neurons = exc_spikes.T
plt.scatter(steps*dt, neurons, s=3)

#%% Output Network, 3 neurons
n = 6
m = 1000

p_neurons = np.array([1,1,1,0,0,0])

a = np.full((n), 0.02, dtype=np.float32)
a[p_neurons < 0.2] = 0.1
d = np.full((n), 8.0, dtype=np.float32)
d[p_neurons < 0.2] = 2.0

w_in = 2*np.random.rand(n,m)-1

# Randomly distribute recurrent connections
w = np.zeros((n,n),  dtype=np.float32)
p_reccur = np.ones((n,n),  dtype=np.float32)
p_reccur[:3,3:] = 0

w[p_reccur < 0.1] = np.random.gamma(2, 0.03, size=w[p_reccur < 0.1].shape)
# Identify inhibitory to excitatory connections (receiving end is in row)
inh_2_exc = np.ix_(p_neurons >= 0.2, p_neurons < 0.2)
# Increase the strength of these connections
w[ inh_2_exc ] = 2* w[ inh_2_exc]

# Only inhibitory neurons have E=-85 mv
e = np.zeros((n), dtype=np.float32)
e[p_neurons<0.2] = -85.0

output_stimuli = (np.array(v_out) > 34)

params = {'no_neurons': 6,
          'dynamics': 'regular_spiking',
          'no_connected_neurons': m,
          'tau': 10.0,
          'W_in': w_in,
          'W': w,
          'E': e,
          'dt': 0.1}

output_network = RecurrentIzhikevichNeuronGroup(params)

sim_params = {'dt': 0.1,
              't_stop': 1500,
              'frate': 0.002,
              'stimuli': output_stimuli}

sim1 = Simulator()
sim1.new_core_syn(output_network, sim_params)

#%% Visualize
v_out = np.array(sim1.results['v_out'])
dt = 0.1
# Split between inhibitory and excitatory
inh_v_out = np.where(p_neurons < 0.2, v_out, 0)
exc_v_out = np.where(p_neurons >= 0.2, v_out, 0)
# Identify spikes
inh_spikes = np.argwhere(inh_v_out == 35.0)
exc_spikes = np.argwhere(exc_v_out == 35.0)
# Display spikes over time

plt.figure()
plt.axis([0, t_stop, -1, n])
plt.title('Output Inhibitory and excitatory spikes')
plt.ylabel('Neurons')
plt.xlabel('Time (msec)')
# Plot inhibitory spikes
steps, neurons = inh_spikes.T

plt.scatter(steps*dt, neurons, s=3)
# Plot excitatory spikes
steps, neurons = exc_spikes.T
plt.scatter(steps*dt, neurons, s=3)