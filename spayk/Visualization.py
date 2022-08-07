#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 19:38:31 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def raster_plot(spike_trains, dt=0.1):
    f = plt.figure()
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    spike_times = []
    for spike_train in spike_trains:
        spike_times.append(np.where(spike_train)[0]*dt)
        
    plt.eventplot(spike_times)
    plt.xlim([0, int(spike_trains.shape[1]*dt)])
    
    ax = f.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    
def plot_voltage_traces(voltages, dt=0.1, selected=None):
    if selected is None:
        selected = np.arange(voltages.shape[0])
    else:
        if isinstance(selected, list):
            selected = np.array(selected)
    
    if len(voltages.shape) == 1:
        voltages = np.expand_dims(voltages, axis=0)
        
    time_array = np.arange(voltages.shape[1])*dt
        
    plt.figure()
    for i, sid in enumerate(selected):
        plt.subplot(selected.shape[0], 1, i+1)
        plt.plot(time_array, voltages[sid])