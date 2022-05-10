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
    def __init__(self, stimuli=None):
        super().__init__()
        self.stimuli = stimuli
        
        self.a = 0.02
        self.b = 0.25
        self.c = -65
        self.d = 6
        self.vt = 30    #mv
        
        self.v = -65
        self.u = 1
        
        self.v_history = [self.v]
        self.u_history = [self.u]
        
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