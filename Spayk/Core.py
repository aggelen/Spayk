#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:41:25 2022

@author: aggelen
"""
import numpy as np

class Simulator:
    def __init__(self):
        self.a = 0.02
        self.b = 0.25
        self.c = -65
        self.d = 6
        self.vt = 30    #mv
    
    def keep_alive(self, organization, settings):
        dt = settings['dt']
        time = np.arange(0,settings['duration'],dt)
        
        #FIXME! Worst soln ever!
        
        for neuron in organization.neurons:
            for t in time:
                neuron.forward(neuron.stimuli.I, dt)
                
        for t in time:
            organization.vs, organization.us, spikes = self.izhikevich_update(organization.vs, 
                                                                              organization.us, 
                                                                              organization.Is,
                                                                              dt)
            organization.keep_log(spikes)
        
        organization.end_of_life()
                
    def izhikevich_update(self, vs, us, Is, dt):
        dv = 0.04*np.square(vs) + 5*vs + 140 - us + Is
        du = self.a*(self.b*vs - us)
        vs = vs + dv*dt
        us = us + du*dt
        
        spikes = vs >= self.vt
        
        us = np.where(vs >= self.vt, us + self.d, us)
        vs = np.where(vs >= self.vt, self.c, vs)

        return vs, us, spikes
            
