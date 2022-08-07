#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:41:25 2022

@author: aggelen
"""
import numpy as np
from tqdm import tqdm

from spayk.Learning import STDP_Engine

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
    
    def new_core(self, tissue, params):
        
        self.results = {'I_in': [],
                        'v_out': []}
        
        dt = params['dt']
        T = params['t_stop']
        steps = range(int(T / dt))
        for step in steps:
            t = step*dt
            
            # We generate a current step of 7 mA between 200 and 700 ms
            if t > 200 and t < 700:
                i_in = 7.0
            else:
                i_in = 0.0
            
            tissue.inject_current(i_in)
            v, u = tissue()
            
            # Store values
            self.results['I_in'].append(i_in)
            self.results['v_out'].append(v)
            
    def new_core_syn(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': [],
                        'presyn_spikes': []}
        
        dt = params['dt']
        T = params['t_stop']
        
        steps = range(int(T / dt))
        for step in tqdm(steps):
            t = step*dt
            
            if 'stimuli' in params.keys():
                p_syn_spike = params['stimuli'][step]
            else:
                if t > 200 and t < 700:
                    # Generate a random matrix
                    r = np.random.uniform(0,1,(tissue.no_connected_neurons))
                    # A synapse has spiked when r is lower than the spiking rate
                    p_syn_spike = r < params['frate'] * params['dt']
                else:
                    # No synapse activity during that period
                    p_syn_spike = np.zeros((tissue.no_connected_neurons), dtype=bool)
            
            if np.sum(p_syn_spike) > 0:
                aykut = 1
            
            tissue.inject_spike_train(p_syn_spike)
            v, u = tissue()
            
            # Store values
            self.results['I_in'].append(tissue.I)
            self.results['v_out'].append(v)
            self.results['presyn_spikes'].append(p_syn_spike)
            
    def new_core_syn_experimental(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': [],
                        'presyn_spikes': [],
                        'weight_means': []}
        
        dt = params['dt']
        T = params['t_stop']
        
        steps = range(int(T / dt))
        for step in tqdm(steps):
            t = step*dt
            
            if 'stimuli' in params.keys():
                p_syn_spike = params['stimuli'][step]
            else:
                if t > 200 and t < 700:
                    # Generate a random matrix
                    r = np.random.uniform(0,1,(tissue.no_connected_neurons))
                    # A synapse has spiked when r is lower than the spiking rate
                    p_syn_spike = r < params['frate'] * params['dt']
                else:
                    # No synapse activity during that period
                    p_syn_spike = np.zeros((tissue.no_connected_neurons), dtype=bool)
            
            tissue.inject_spike_train(p_syn_spike)
            v, u = tissue()
            
            # Store values
            self.results['I_in'].append(tissue.I)
            self.results['v_out'].append(v)
            self.results['presyn_spikes'].append(p_syn_spike)
            # self.results['weight_means'].append([tissue.W.mean(), tissue.W_in.mean()])
            self.results['weight_means'].append(tissue.W_in.mean())
            
    def new_core_syn_stdp(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': []}
        
        dt = params['dt']
        T = params['t_stop']
        
        steps = range(int(T / dt))
        
        w_prev = tissue.W_in
        delta_weights = np.zeros((int(T / dt), tissue.no_connected_neurons))
    
        
        for step in steps:
            t = step*dt
            
            if 'stimuli' in params.keys():
                p_syn_spike = params['stimuli'][step]
            else:
                if t > 200 and t < 700:
                    # Generate a random matrix
                    r = np.random.uniform(0,1,(tissue.no_connected_neurons))
                    # A synapse has spiked when r is lower than the spiking rate
                    p_syn_spike = r < params['frate'] * params['dt']
                else:
                    # No synapse activity during that period
                    p_syn_spike = np.zeros((tissue.no_connected_neurons), dtype=bool)
            
            tissue.inject_spike_train(p_syn_spike)
            v, u = tissue()
            
            # Store values
            self.results['I_in'].append(tissue.I)
            self.results['v_out'].append(v)
            
            w_next = tissue.W_in
            delta_weights[step,:] = w_next - w_prev
            w_prev = w_next
            
        self.results['delta_weights'] = delta_weights
        
    def keep_alive(self, organizations, settings):
        dt = settings['dt']
        time = np.arange(0,settings['duration'],dt)
        
        if 'synaptic_plasticity' in settings.keys():
            self.synaptic_plasticity = settings['synaptic_plasticity']
        else:
            raise Exception('synaptic_plasticity status must be set!')
        
        #FIXME! Worst soln ever!
        # for neuron in organization.neurons:
        #     for t in time:
        #         neuron.forward(neuron.stimuli.I, dt)
                
        for t_id in tqdm(range(time.shape[0])):
            t = time[t_id]
            for organization in organizations:
                self.izhikevich_update(organization, t, dt)
                
        for organization in organizations:
            organization.end_of_life()
                
    def izhikevich_update(self, organization, t, dt):
        vs, us, dMat = organization.vs, organization.us, organization.dynamics_matrix
        
        Is = organization.calculate_Is(t, self.synaptic_plasticity)
        
        a,b,c,d,vt = dMat[:,0],dMat[:,1],dMat[:,2],dMat[:,3],dMat[:,4]
        
        dv = 0.04*np.square(vs) + 5*vs + 140 - us + Is
        du = a*(b*vs - us)
        vs = vs + dv*dt
        us = us + du*dt

        spikes = np.greater_equal(vs,vt)
        
        #FIXME: delete org name
        # if any(spikes):
        if organization.stdp_status and organization.name=='network' and t > 0.0:
            organization.LTD_update(spikes, dt)
    
        us = np.where(spikes, us + d, us)
        vs = np.where(spikes, c, vs)
        
        organization.vs = vs
        organization.us = us
        organization.keep_log(spikes, Is)
        
    def integrate_and_fire(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': []}
        
        dt = params['dt']
        T = params['t_stop']
        
        steps = range(int(T / dt))
        for step in tqdm(steps):
            p_syn_spikes = params['stimuli'][step]
            
            v = tissue(p_syn_spikes)
            self.results['v_out'].append(v)
            
    def integrate_and_fire_stdp(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': [],
                        'delta_w': [],
                        'mean_w': []}
        
        dt = params['dt']
        T = params['t_stop']
        steps = range(int(T / dt))
        w_prev = tissue.w
        
        for step in tqdm(steps):
            p_syn_spikes = params['stimuli'][step]
            
            v = tissue(p_syn_spikes)
            self.results['v_out'].append(v)
            self.results['mean_w'].append(tissue.w.mean())
            
            w_next = tissue.w
            delta_weights = w_next - w_prev
            w_prev = w_next
            self.results['delta_w'].append(delta_weights)
            
            