#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:24:08 2022

@author: aggelen
"""
import numpy as np

class Synapse:
    def __init__(self):
       self.channels = {}
        
    def create_channel(self, params, descriptor, model='single_decay'):
        self.channels[descriptor] = Channel(params, descriptor, model)
    
    def calculate_syn_current(self, t, channel, u, Esyn):
        """
        Calculate synaptic current
        """
        return channel.g_syn(t)*(u - Esyn)
    
class Channel:
    def __init__(self, params, descriptor, model):
        self.model = model
        self.g_syn0 = params['g_syn0']
        self.tau = params['tau']
        self.tf = params['tf']
        
        if model == 'double_decay':
            self.tau_fast = params['tau_fast']
            self.tau_slow = params['tau_slow']
            self.tau_rise = params['tau_rise']
            self.a = params['a']
        
        self.descriptor = descriptor
    
    def g_syn(self, t):
        if self.model == 'single_decay':
            return self.single_decay(t)
        elif self.model == 'double_decay':
            return self.double_decay(t)
        else:
            raise NotImplementedError()
            
    def single_decay(self, t):
        return self.g_syn0*np.exp(-(t-self.tf)/self.tau)*np.heaviside(t-self.tf, 0.5)
    
    def double_decay(self, t):
        rise = 1-np.exp(-(t-self.tf)/self.tau_rise)
        fast = self.a*np.exp(-(t-self.tf)/self.tau_fast)
        slow = (1-self.a)*np.exp(-(t-self.tf)/self.tau_slow)
        return self.g_syn0*rise*(fast+slow)*np.heaviside(t-self.tf, 0.5)