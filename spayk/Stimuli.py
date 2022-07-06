#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:49:06 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

class ConstantCurrentSource:
    def __init__(self, params):
        ts, te, self.t_stop, amp = params['t_cur_start'], params['t_cur_end'], params['t_stop'], params['amplitudes']
        self.dt = params['dt']
        self.steps = np.arange(int(self.t_stop / self.dt))
        
        # signals = np.multiply(np.ones((ts.size, self.steps.size)), amp[:, np.newaxis])
        signals = np.zeros((ts.size, self.steps.size))
        ind_s, ind_e = ts/self.dt, te/self.dt
        
        for i in range(ts.size):
            signals[i, int(ind_s[i]):int(ind_e[i])] = amp[i]
            
        self.currents = signals
        self.current_step = 0
        self.source_type = 'current'
        
    def plot(self):
        plt.figure()
        plt.plot(np.arange(self.currents.shape[1])*self.dt, self.currents.T)
        plt.title('Injected Currents')
        plt.xlabel('Time (ms)')
        plt.ylabel('Current ()')
        
    def I(self):
        I = self.currents[:,self.current_step]
        self.current_step += 1 
        return I

class SpikeTrain:
    def __init__(self):
        pass
    
    def current_spikes(self):
        spikes = self.spikes[:,self.current_step]
        self.current_step += 1 
        return spikes
        
    def raster_plot(self, color_array=None, first_n=None):
        f = plt.figure()
        plt.title('Raster Plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        # spike_times = []
        
        mean_spike_rate = np.sum(self.spikes,1).mean() / (self.t_stop/1000)
        print('Output Mean Spike Rate: {}'.format(mean_spike_rate))
        
        if first_n is not None:
            spike_loc = np.argwhere(self.spikes[:first_n,:])
        else:
            spike_loc = np.argwhere(self.spikes)
        
        sns.set()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if color_array is not None:
            # c = plt.cm.Set1(color_array[spike_loc[:,0]])
            c = []
            for k in color_array[spike_loc[:,0]]:
                c.append(colors[int(k)])
        else:
            c = colors[0]
            
        plt.scatter(spike_loc[:,1]*self.dt, spike_loc[:,0], s=3, color=c)
    

class ExternalSpikeTrain(SpikeTrain):
    def __init__(self, dt, t_stop, no_neurons, spike_train):
        super().__init__()
        self.dt = dt
        self.t_stop = t_stop
        self.no_neurons = no_neurons
        self.steps = np.arange(int(t_stop / dt))
        
        self.spikes = spike_train
        
        self.source_type = 'spike_train'
        self.current_step = 0
        
        self.mean_spike_rate = np.sum(self.spikes,1).mean() / (self.t_stop/1000)
        print('Spike Train Mean Spike Rate: {}'.format(self.mean_spike_rate))
        
    def current_spikes(self):
        if self.spikes.shape.__len__() == 1:
            self.spikes = np.expand_dims(self.spikes, axis=0)
        spikes = self.spikes[:,self.current_step]
        self.current_step += 1 
        return spikes
        
class PoissonSpikeTrain(SpikeTrain):
    def __init__(self, dt, t_stop, no_neurons, spike_rates):
        super().__init__()
        #dt in ms
        self.dt = dt
        self.t_stop = t_stop
        self.no_neurons = no_neurons
        self.steps = np.arange(int(t_stop / dt))
        prob = np.random.uniform(0, 1, (self.steps.size, self.no_neurons))
        self.spikes = np.less(prob, np.array(spike_rates)*dt*1e-3).T
        self.source_type = 'spike_train'
        self.current_step = 0
        
        self.mean_spike_rate = np.sum(self.spikes,1).mean() / (self.t_stop/1000)
        print('Spike Train Mean Spike Rate: {}'.format(self.mean_spike_rate))
          
class ExternalCurrentSignal:
    def __init__(self, signal):
        self.signal = signal
        self.idx = 0
        
    def I(self):
        I = self.signal[self.idx]
        self.idx += 1
        return I