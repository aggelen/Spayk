#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:49:06 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    
class PoissonSpikeTrain:
    def __init__(self, dt, t_stop, no_neurons, spike_rates):
        #dt in ms
        self.dt = dt
        self.t_stop = t_stop
        self.no_neurons = no_neurons
        self.steps = np.arange(int(t_stop / dt))
        prob = np.random.uniform(0, 1, (self.steps.size, self.no_neurons))
        self.spikes = np.less_equal(prob, np.array(spike_rates)*dt*1e-3).T
        self.source_type = 'spike_train'
        self.current_step = 0
        
    def current_spikes(self):
        spikes = self.spikes[:,self.current_step]
        self.current_step += 1 
        return spikes
        
    def raster_plot(self):
        f = plt.figure()
        plt.title('Raster Plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        spike_times = []
        
        for spike_train in self.spikes:
            spike_times.append(np.where(spike_train)[0]*self.dt)
            
        plt.eventplot(spike_times, color='k')
        plt.xlim([0, int(self.spikes.shape[1]*self.dt)])
        
        ax = f.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    
# class ConstantCurrentSource:
#     def __init__(self, mA):
#         self.mA = mA
    
#     def I(self):
#         return self.mA
    
# class ExternalCurrentSignal:
#     def __init__(self, signal):
#         self.signal = signal
#         self.idx = 0
        
#     def I(self):
#         I = self.signal[self.idx]
#         self.idx += 1
#         return I
        
# A class that generates random spike trains
class SpikeTrains(object):
    
    def __init__(self, n_syn, r_min=0.0, r_max=90.0, r=None, s_max=1800, ds_max=360, s=None, auto_vrate=True, delta_max=0, dt=0.1):
        
        self.dt = dt*1e-3   #dt in ms
        # Number of synapses
        self.n_syn = n_syn
        # Minimum and maximum spiking rate (in Hz)
        self.r_min = r_min
        self.r_max = r_max
        # Spiking rate for each synapse (in Hz)
        if r is None:
            self.r = np.random.uniform(self.r_min, self.r_max, size=(n_syn))
        else:
            self.r = r
        # Rate variation parameters
        self.s_max = s_max
        self.ds_max = ds_max
        # Rate variation
        if s is None:
            self.s = np.random.uniform(-self.s_max, self.s_max, size=(self.n_syn))
        else:
            self.s = s
        # Automatically apply rate variation when
        self.auto_vrate = auto_vrate
        # Maximum time between two spikes on each synapse (0 means no maximum) in ms
        self.delta_max = delta_max

        # Memory of spikes
        self.spikes = None
    
    # Generate new spikes for the specified time interval (in ms)
    # The new spikes are added to the existing spike trains.
    # The method returns only the new set of spikes
    def add_spikes(self, t):
        
        for step in range(t):
            # Draw a random number for each synapse
            x = np.random.uniform(0,1, size=(self.n_syn))
            # Each synapse spikes if the drawn number is lower than the probablity
            # given by the integration of the rate over one millisecond
            # spikes = x < self.r * 1e-3
            spikes = np.less_equal(x, self.r*self.dt)
            # Keep a memory of our spikes
            if self.spikes is None:
                self.spikes = np.array([spikes])
            else:
                if self.delta_max > 0:
                    # We force each synapse to spike at least every delta_max ms
                    if self.spikes.shape[0] < self.delta_max - 1:
                        # At the beginning of the trains, we try to 'fill' as much holes
                        # as possible to avoid a 'wall of spikes' when we reach delta_max.
                        # For each synapse, count non-zero items
                        n_spikes = np.count_nonzero(self.spikes, axis=0)
                        # Draw a random number for each synapse 
                        r = np.random.uniform(0.0, 1.0, size=self.n_syn)
                        # The closer we get to delta_max, the higher probability we have to force a spike
                        forced_spikes = r < step * 1.0/self.delta_max
                        # Modify our random vector of spikes for synapse that did not spike
                        spikes = np.where(n_spikes > 0, spikes, spikes | forced_spikes)
                    else:
                        # Get the last delta_max -1 spike trains
                        last_spikes = self.spikes[-(self.delta_max - 1):,:]
                        # For each synapse, count non-zero items
                        n_spikes = np.count_nonzero(last_spikes, axis=0)
                        # Modify spikes to force a spike on synapses where the spike count is zero
                        spikes = np.where(n_spikes > 0, spikes, True)
                # Store spikes
                self.spikes = np.append(self.spikes, [spikes], axis=0)
            if self.auto_vrate:
                self.change_rate()

        return self.spikes[-t:,:]
    
    # Format a list of spike indexes
    def get_spikes(self):
        
        real_spikes = np.argwhere(self.spikes > 0)
        # We prefer having spikes in the range [1..n_syn]
        spike_index = real_spikes[:,1] + 1
        spike_timings = real_spikes[:,0]
        
        return spike_timings, spike_index
    
    # Change rate, applying the specified delta in Hz
    def change_rate(self, delta=None):

        # Update spiking rate
        if delta is None:
            delta = self.s*self.dt
        self.r = np.clip( self.r + delta, self.r_min, self.r_max)
        # Update spiking rate variation
        ds = np.random.uniform(-self.ds_max, self.ds_max, size=(self.n_syn))
        self.s = np.clip( self.s + ds, -self.s_max, self.s_max)