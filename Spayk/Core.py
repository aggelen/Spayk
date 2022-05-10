#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:41:25 2022

@author: aggelen
"""
import numpy as np

class Simulator:
    def __init__(self):
        pass
    
    def keep_alive(self, organization, settings):
        dt = settings['dt']
        time = np.arange(0,settings['duration'],dt)
        
        #FIXME! Worst soln ever!
        
        for neuron in organization.neurons:
            for t in time:
                neuron.forward(neuron.stimuli.I, dt)
                
            