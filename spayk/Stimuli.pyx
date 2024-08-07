#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:45:25 2024

@author: gelenag
"""

import numpy as np
import matplotlib.pyplot as plt
np.import_array()

ctypedef np.float64_t real
ctypedef unsigned int number
ctypedef char* cstring

cdef class Array:
    cdef real* data

    def __setitem__(self, number index, real value):
        self.data[index] = value

    def __getitem__(self, number index):
        return self.data[index]
    
#%% Dummy for codegen
cdef class Activity:
    
    cdef cstring activity_type
    
    def __init__(self, cstring activity_type):
        self.activity_type = activity_type
        
class PoissonActivity(Activity):
    def __init__(self, number no_neurons, np.ndarray[real, ndim=1] firing_rates, np.ndarray[real, ndim=1] time_params, cstring group_label):
        super().__init__('poisson')
        self.no_neurons = no_neurons
        self.firing_rates = firing_rates
        self.time_params = time_params
        self.group_label = group_label
        
        self.neuron_labels = [str(group_label)+str(i) for i in range(no_neurons)]

#%% Actual classes
class SpikeTrain:
    def __init__(self, spikes, time_params):
        self.spikes = spikes
        self.time_params = time_params
        
    def raster_plot(self, color_array=None, first_n=None, title=None, title_pad=0):
        ax = plt.figure().gca()
        
        
        if title is not None:
            plt.title(title, pad=title_pad)
        else:
            plt.title('Raster Plot', pad=title_pad)
            
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')

        if first_n is not None:
            spike_loc = np.argwhere(self.spikes[:first_n,:])
        else:
            spike_loc = np.argwhere(self.spikes)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if color_array is not None:
            # c = plt.cm.Set1(color_array[spike_loc[:,0]])
            c = []
            for k in color_array[spike_loc[:,0]]:
                c.append(colors[int(k)])
        else:
            c = colors[0]

        plt.scatter(spike_loc[:,1], spike_loc[:,0], s=2, color=c)
        
        yticks = range(min(spike_loc[:,0]), max(spike_loc[:,0]) + 1)
        ax.set_yticks(yticks)
        
        ax.set_xlim([0,10000])
        
    def firing_rates(self):
        t_start, t_stop, dt = self.time_params
        total_time = t_stop - t_start
        fring_rates = np.sum(self.spikes,1) / total_time
        
        print('Fring Rates: {}'.format(fring_rates))
      
#%%    
class PoissonSpikeTrain(SpikeTrain):
    def __init__(self, no_neurons, firing_rates, time_params):
        spikes = poisson_spike_generator(no_neurons, firing_rates, time_params)
        super().__init__(spikes, time_params)
    
def poisson_spike_generator(no_neurons, firing_rates, time_params):
    """
    
    :param no_neurons: Number of Neurons
    :type no_neurons: int
    :param firing_rates: Firing rate for each neuron, 1d array
    :type firing_rates: ndarray
    :param time_params: Time parameters for generation, ie. t_start, t_stop, delta_time in sec
    :type time_params: list
    :return: Generated spikes
    :rtype: ndarray

    """
    
    t_start, t_stop, dt = time_params
    
    time_array = np.arange(t_start, t_stop, dt)
    prob = np.random.uniform(0, 1, (time_array.size, no_neurons))
    spikes = np.less(prob, np.array(firing_rates)*dt)
    
    return spikes.T

#%% 
class ExternalSpikeTrain(SpikeTrain):
    def __init__(self, spikes, time_params):
        super().__init__(spikes, time_params)

