#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:41:25 2022

@author: aggelen
"""
import numpy as np

class Simulator:
    def __init__(self):
        """
        Simulator module!
        """
        # self.a = 0.02
        # self.b = 0.25
        # self.c = -65
        # self.d = 6
        # self.vt = 30    #mv
        pass
    
    def keep_alive(self, organization, settings):
        dt = settings['dt']
        time = np.arange(0,settings['duration'],dt)
        
        #FIXME! Worst soln ever!
        
        for neuron in organization.neurons:
            for t in time:
                neuron.forward(neuron.stimuli.I, dt)
                
        for t in time:
            self.izhikevich_update(organization, dt)
            # organization.keep_log(spikes)
        
        organization.end_of_life()
                
    def izhikevich_update(self, organization, dt):
        vs,us,Is, dMat = organization.vs, organization.us, organization.Is, organization.dynamics_matrix
        a,b,c,d,vt = dMat[:,0],dMat[:,1],dMat[:,2],dMat[:,3],dMat[:,4]
        
        dv = 0.04*np.square(vs) + 5*vs + 140 - us + Is
        du = a*(b*vs - us)
        vs = vs + dv*dt
        us = us + du*dt

        spikes = np.greater_equal(vs,vt)
        
        us = np.where(spikes, us + d, us)
        vs = np.where(spikes, c, vs)
        
        organization.vs = vs
        organization.us = us
        organization.keep_log(spikes)
            
