#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:57:30 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Tissue:
    def __init__(self):
        self.neurons = []
        
        self.a = 0.02
        self.b = 0.25
        self.c = -65
        self.d = 6
        self.vt = 30    #mv
        self.v0 = -65
        
        self.vs = []
        self.us = []
        self.Is = []
        
        self.log_v = []
        self.log_u = []
        self.log_spikes = []
        
    def add(self, neurons):
        for n in neurons:
            self.neurons.append(n)
            
            self.vs.append(self.v0)
            self.us.append(self.b*self.v0)
            self.Is.append(n.stimuli.I)
            
    def embody(self):
        self.vs = np.array(self.vs)
        self.us = np.array(self.us)
        self.Is = np.array(self.Is)
        
    def keep_log(self, spikes):
        self.log_v.append(self.vs)
        self.log_u.append(self.us)
        self.log_spikes.append(spikes)
        
    def end_of_life(self):
        self.log_v = np.array(self.log_v)
        self.log_u = np.array(self.log_u)
        self.log_spikes = np.array(self.log_spikes, dtype=bool).T
        
    def plot_membrane_potential_of(self, neuron_id, dt=0.1, hold_on=False):
        time = np.arange(self.log_v.shape[0])*dt
        if not hold_on:
            plt.figure()
        plt.plot(time, self.log_v[:, neuron_id])
        plt.xlabel('Time (ms)')
        plt.ylabel('mV')
        plt.title('Membrane Potential of Neuron#{}'.format(neuron_id))
        plt.grid()
        plt.xlim([0, int(self.log_v.shape[0]*dt)])
        
    def raster_plot(self, dt=0.1):
        f = plt.figure('Raster Plot')
        plt.title('Raster Plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        spike_times = []
        for spike_train in self.log_spikes:
            spike_times.append(np.where(spike_train)[0]*dt)
        plt.eventplot(spike_times)
        plt.xlim([0, int(self.log_spikes.shape[1]*dt)])
        
        ax = f.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    