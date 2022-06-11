#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:03:39 2022

@author: aggelen
"""

from spayk.Nerves import IzhikevichNeuronGroup, SynapticIzhikevichNeuronGroup, RecurrentIzhikevichNeuronGroup
from spayk.Core import Simulator

# #%% Create Model
# params = {'no_neurons': 1,
#           'dynamics': 'regular_spiking'}
# network = IzhikevichNeuronGroup(params)

# #%% Simulate
# sim_params = {'dt': 0.1,
#               't_stop': 1000}

# sim0 = Simulator()
# sim0.new_core(network, sim_params)

# #%% Visualize
# import matplotlib.pyplot as plt
# import numpy as np
# plt.plot(np.arange(len(sim0.results['v_out']))*0.1, sim0.results['v_out'])


# #%% 2. Create Model
# params = {'no_neurons': 1,
#           'dynamics': 'regular_spiking',
#           'no_connected_neurons': 100,
#           'tau': 10.0}
# network = SynapticIzhikevichNeuronGroup(params)

# #%% Simulate
# sim_params = {'dt': 0.1,
#               't_stop': 1000,
#               'frate': 0.002}

# sim0 = Simulator()
# sim0.new_core_syn(network, sim_params)

# #%% Visualize
# import matplotlib.pyplot as plt
# import numpy as np
# plt.plot(np.arange(len(sim0.results['v_out']))*0.1, sim0.results['v_out'])

# plt.figure()
# plt.plot(np.arange(len(sim0.results['I_in']))*0.1, sim0.results['I_in'])

#%% 3. Create Model
# import numpy as np
# n = 1000
# m = 100
# p_neurons = np.random.uniform(0,1,(n))

# a = np.full((n), 0.02, dtype=np.float32)
# a[p_neurons < 0.2] = 0.1
# d = np.full((n), 8.0, dtype=np.float32)
# d[p_neurons < 0.2] = 2.0

# # Randomly connect 10% of the neurons to the input synapses
# p_syn = np.random.uniform(0,1,(n,m))
# w_in = np.zeros((n,m), dtype=np.float32)
# w_in[ p_syn < 0.1 ] = 0.07

# params = {'no_neurons': 1,
#           'dynamics': 'regular_spiking',
#           'no_connected_neurons': m,
#           'tau': 10.0,
#           'A': a,
#           'D': d,
#           'W_in': w_in}

# network = SynapticIzhikevichNeuronGroup(params)

# #%% Simulate
# sim_params = {'dt': 0.1,
#               't_stop': 1000,
#               'frate': 0.002}

# sim0 = Simulator()
# sim0.new_core_syn(network, sim_params)

# #%% Visualize
# import matplotlib.pyplot as plt
# v_out = sim0.results['v_out']
# dt = 0.1
# # Split between inhibitory and excitatory
# inh_v_out = np.where(p_neurons < 0.2, v_out, 0)
# exc_v_out = np.where(p_neurons >= 0.2, v_out, 0)
# # Identify spikes
# inh_spikes = np.argwhere(inh_v_out == 35.0)
# exc_spikes = np.argwhere(exc_v_out == 35.0)
# # Display spikes over time

# plt.axis([0, 1000, 0, n])
# plt.title('Inhibitory and excitatory spikes')
# plt.ylabel('Neurons')
# plt.xlabel('Time (msec)')
# # Plot inhibitory spikes
# steps, neurons = inh_spikes.T

# plt.scatter(steps*dt, neurons, s=3)
# # Plot excitatory spikes
# steps, neurons = exc_spikes.T
# plt.scatter(steps*dt, neurons, s=3)


# 1000 neurons with recurrent connections
import numpy as np
n = 1000
m = 100
p_neurons = np.random.uniform(0,1,(n))

a = np.full((n), 0.02, dtype=np.float32)
a[p_neurons < 0.2] = 0.1
d = np.full((n), 8.0, dtype=np.float32)
d[p_neurons < 0.2] = 2.0

# Randomly connect 10% of the neurons to the input synapses
p_syn = np.random.uniform(0,1,(n,m))
w_in = np.zeros((n,m), dtype=np.float32)
w_in[ p_syn < 0.1 ] = 0.07

# Randomly distribute recurrent connections
w = np.zeros((n,n),  dtype=np.float32)
p_reccur = np.random.uniform(0,1,(n,n))
w[p_reccur < 0.1] = np.random.gamma(2, 0.003, size=w[p_reccur < 0.1].shape)
# Identify inhibitory to excitatory connections (receiving end is in row)
inh_2_exc = np.ix_(p_neurons >= 0.2, p_neurons < 0.2)
# Increase the strength of these connections
w[ inh_2_exc ] = 2* w[ inh_2_exc]

# Only inhibitory neurons have E=-85 mv
e = np.zeros((n), dtype=np.float32)
e[p_neurons<0.2] = -85.0

params = {'no_neurons': 1,
          'dynamics': 'regular_spiking',
          'no_connected_neurons': m,
          'tau': 10.0,
          'A': a,
          'D': d,
          'W_in': w_in,
          'W': w,
          'E': e}

network = RecurrentIzhikevichNeuronGroup(params)

#%% Simulate
sim_params = {'dt': 0.1,
              't_stop': 1000,
              'frate': 0.002}

sim0 = Simulator()
sim0.new_core_syn(network, sim_params)

#%% Visualize
import matplotlib.pyplot as plt
v_out = sim0.results['v_out']
dt = 0.1
# Split between inhibitory and excitatory
inh_v_out = np.where(p_neurons < 0.2, v_out, 0)
exc_v_out = np.where(p_neurons >= 0.2, v_out, 0)
# Identify spikes
inh_spikes = np.argwhere(inh_v_out == 35.0)
exc_spikes = np.argwhere(exc_v_out == 35.0)
# Display spikes over time

plt.axis([0, 1000, 0, n])
plt.title('Inhibitory and excitatory spikes')
plt.ylabel('Neurons')
plt.xlabel('Time (msec)')
# Plot inhibitory spikes
steps, neurons = inh_spikes.T

plt.scatter(steps*dt, neurons, s=3)
# Plot excitatory spikes
steps, neurons = exc_spikes.T
plt.scatter(steps*dt, neurons, s=3)