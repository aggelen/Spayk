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
                
        self.vs = []
        self.us = []
        self.Is = []
        
        self.dynamics_matrix = []
        self.neuron_modes = []
        
        self.log_v = []
        self.log_u = []
        self.log_spikes = []
        
    def add(self, neurons):
        for n in neurons:
            self.neurons.append(n)
            
            #FIXME! Initialization
            self.vs.append(-65)
            self.us.append(-13)
            self.Is.append(n.stimuli.I)
            self.dynamics_matrix.append([n.a,n.b,n.c,n.d,n.vt])
            self.neuron_modes.append(n.mode)
            
    def embody(self):
        self.vs = np.array(self.vs)
        self.us = np.array(self.us)
        self.Is = np.array(self.Is)
        self.dynamics_matrix = np.array(self.dynamics_matrix)
    
    def keep_log(self, spikes):
        self.log_v.append(self.vs)
        self.log_u.append(self.us)
        self.log_spikes.append(spikes)
        
    def end_of_life(self):
        self.log_v = np.array(self.log_v)
        self.log_u = np.array(self.log_u)
        self.log_spikes = np.array(self.log_spikes, dtype=bool).T
        
    def plot_membrane_potential_of(self, neuron_id, dt=0.1, hold_on=False, color=None):
        time = np.arange(self.log_v.shape[0])*dt
        if not hold_on:
            plt.figure()
        if color is not None:
            plt.plot(time, self.log_v[:, neuron_id], color)
        else:
            plt.plot(time, self.log_v[:, neuron_id])
        plt.xlabel('Time (ms)')
        plt.ylabel('mV')
        plt.title('Membrane Potential of Neuron#{}'.format(neuron_id))
        plt.grid()
        plt.xlim([0, int(self.log_v.shape[0]*dt)])
        
    def raster_plot(self, dt=0.1):
        # color_list = [ "red", "blue", "green", "yellow", "purple", "orange", "white", "black" ]
        color_list = plt.get_cmap('tab10')
        colors = [color_list(mode) for mode in self.neuron_modes]
        
        labels = ['regular_spiking',
                  'intrinsically_bursting', 
                  'chattering', 
                  'fast_spiking', 
                  'thalamo_cortical',
                  'resonator',
                  'low_threshold_spiking']
        
        f = plt.figure('Raster Plot')
        plt.title('Raster Plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        spike_times = []
        for spike_train in self.log_spikes:
            spike_times.append(np.where(spike_train)[0]*dt)
        ep = plt.eventplot(spike_times, colors=colors)
        plt.xlim([0, int(self.log_spikes.shape[1]*dt)])
        
        ax = f.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.legend(labels, colors)