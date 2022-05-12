#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:17:14 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self):
        pass
    
class SingleIzhikevichNeuron(Neuron):
    neuron_dynamics = {'regular_spiking': 0,
                       'intrinsically_bursting': 1,
                       'chattering': 2,
                       'fast_spiking': 3,
                       'thalamo_cortical': 4,
                       'resonator': 5,
                       'low_threshold_spiking': 6}
    
    def __init__(self, stimuli=None, dynamics='regular_spiking'):
        super().__init__()
        self.stimuli = stimuli
        
        # self.a = 0.02
        # self.b = 0.2
        # self.c = -65
        # self.d = 6
        
        self.dynamics_selector(dynamics)
        self.mode = self.neuron_dynamics[dynamics]
        
        self.vt = 30    #mv
        
        self.v = -65
        self.u = self.b * self.v
        
        self.v_history = [self.v]
        self.u_history = [self.u]
        
    def dynamics_selector(self, mode):
        if mode == 'regular_spiking':
            self.a = 0.02
            self.b = 0.25
            self.c = -65
            self.d = 8
        elif mode == 'intrinsically_bursting':
            self.a = 0.02
            self.b = 0.2
            self.c = -55
            self.d = 4
        elif mode == 'chattering':
            self.a = 0.02
            self.b = 0.2
            self.c = -50
            self.d = 2
        elif mode == 'fast_spiking':
            self.a = 0.1
            self.b = 0.2
            self.c = -65
            self.d = 2
        elif mode == 'thalamo_cortical':
            self.a = 0.02
            self.b = 0.25
            self.c = -65
            self.d = 0.05
        elif mode == 'resonator':
            self.a = 0.1
            self.b = 0.25
            self.c = -65
            self.d = 8
        elif mode == 'low_threshold_spiking':
            self.a = 0.02
            self.b = 0.2
            self.c = -65
            self.d = 2
        else:
            raise Exception('Invalid Dynamics for Izhikevich Model')       
        
    def forward(self, I, dt):
        dv = 0.04*self.v**2 + 5*self.v + 140 - self.u + I
        du = self.a * (self.b*self.v - self.u)
        
        self.v = self.v + dv*dt
        self.u = self.u + du*dt
        
        if self.v >= self.vt:
            self.v = self.c
            self.u = self.u + self.d

        self.v_history.append(self.v)
        self.u_history.append(self.u)
        
    def plot_v(self, dt):
        time = np.arange(len(self.v_history))*dt
        plt.figure()
        plt.plot(time, self.v_history)
        plt.xlabel('Time (ms)')
        plt.ylabel('mV')