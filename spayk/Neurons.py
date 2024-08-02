#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:30:19 2024

@author: gelenag
"""
import numpy as np
import matplotlib.pyplot as plt

class LIFGroup:
    def __init__(self, no_neurons, group_label, params):
        self.group_label = group_label
        self.no_neurons = no_neurons
        self.params = params
        
        # table with cols AMPA AMPAEXT NMDA GABA
        self.neuron_channel_assoc_table = np.zeros((self.no_neurons, 4))
        
        self.neuron_labels = [str(group_label)+str(i) for i in range(no_neurons)]
        
    def firing_rate_curve(self):
        dt = 0.1e-3
        time_array = np.arange(0,0.5,dt)
        I_syn_array = np.arange(4.5e-10, 1e-9, 1e-12)
        # I_syn_array = [4.95e-10, 5.0e-10, 5.5e-10]
        firing_rates = []
        for I_syn in I_syn_array:
            v = self.params['VR']
            t_rest = 0
            output_spikes = []
            for t in time_array:
                
                is_in_rest = np.greater(t_rest, 0.0)
                
                t_rest = np.where(is_in_rest, t_rest-dt, t_rest)
            
                dv = (-self.params['GL']*(v - self.params['VL']) + I_syn) / self.params['CM']
                integrated_v = v + dv*dt
                
                v = np.where(np.logical_not(is_in_rest), integrated_v, v)
            
                is_fired = np.greater_equal(v, self.params['VT'])
                
                v = np.where(is_fired, self.params['VR'], v)
                t_rest = np.where(is_fired, self.params['TREF'], t_rest)
        
                output_spikes.append(np.copy(is_fired))
                
            firing_rates.append(np.sum(output_spikes)*2)
        
        str_tobe_saved = """"""
        for i in range(len(firing_rates)):
            str_tobe_saved += "fr_{}@{}_A\n".format(firing_rates[i], I_syn_array[i])
        
        with open("neuron_firing_rate_data.txt", "w") as text_file:
            text_file.write(str_tobe_saved)
            
        plt.figure()
        plt.plot(I_syn_array, firing_rates)
        plt.xlabel('Isyn')
        plt.ylabel('Firing Rate (Hz)')
