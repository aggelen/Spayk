#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:41:25 2022

@author: aggelen
"""
import numpy as np
from tqdm import tqdm
from collections import defaultdict 

from spayk.Configurations import SynapseConfigurator
from spayk.Synapses import COBASynapses
from spayk.Visualization import NetworkVisualizer

# from spayk.Learning import STDP_Engine

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
            
#%% Forward Time Step Based Simulator
# Experimental step-based simulator. It may run very slowly.

class NeuralNetwork:
    def __init__(self, dt):
        self.dt = dt
        self.neuron_id_counter = 0
        
        self.neuron_list = []
        self.external_sources = []
        
        self.visualizer = NetworkVisualizer()
        
    def add_neuron(self, neurons):
        for neuron in neurons:
            self.neuron_list.append(neuron)
            self.neuron_id_counter += 1
            
            self.visualizer.add_node(neuron.visual_style, neuron.visual_position)
            
    def add_externals(self, externals):
        for ext in externals:
            self.external_sources.append(ext)
            
            
    def add_connection(self, connection):
        s = connection.split(";")
        for cnn in s[1:]:
            ch, target = cnn.split('@')
            
            current_config = SynapseConfigurator(self.dt)
            
            if ch.strip() == "EXT_AMPA":
                target_stim_id = int(target[1:])
                current_config.create_external_AMPA_channel(self.external_sources[target_stim_id])
                
            if ch.strip() == "REC_AMPA":
                target_neuron_id = int(target[1:])
                current_config.create_recurrent_AMPA_channel(self.neuron_list[target_neuron_id])
                
            synapses = COBASynapses(current_config.generate_config())
            self.neuron_list[int(s[0][1:])].synapses = synapses
            
    def __no_neurons__(self):
        return len(self.neuron_list)

#%% Logger
class Logger:
    def __init__(self):
        self.neuron_v_traces = None
        
    def generate_traces(self, no_neurons, no_time_steps):
        self.neuron_v_traces = np.empty((no_neurons, no_time_steps))
        self.neuron_I_traces = np.empty((no_neurons, no_time_steps))
        
    def update_neuron_traces(self, neuron_id, time_step, v, I):
        self.neuron_v_traces[neuron_id, time_step] = v
        self.neuron_I_traces[neuron_id, time_step] = I          #total synaptic current
    
#%% Simulator

class DiscreteTimeSimulator:
    def __init__(self, dt):
        self.dt = dt
        self.neural_network = None
        
        self.simulation_log = Logger()
        
        #FIXME : need better logger
        self.v_logs = []
        self.I_logs = []
        
    def configure_neural_network(self, neural_network):
        self.neural_network = neural_network
        
    def create_time(self, t_stop):
        return np.arange(0, t_stop, self.dt)
        
    def keep_alive(self, t_stop):
        self.time_hist = self.create_time(t_stop)
        self.simulation_log.time_hist = self.time_hist
        
        self.simulation_log.generate_traces(self.neural_network.__no_neurons__(), len(self.time_hist))

        
        if self.neural_network is not None:
            # Main Loop
            for time_step, t in enumerate(self.time_hist):
                neuron_v_logs = []
                neuron_I_logs = []
                for neuron_id, neuron in enumerate(self.neural_network.neuron_list):
                    #each neuron has a pre-configured channel stack, calculate each channels current
                    
                    last_postsynaptic_spike = neuron.spikes[-1] if len(neuron.spikes) else 0
                    neuron.synapses.spiked = last_postsynaptic_spike
                    
                    I_syn = neuron.calculate_synaptic_current(time_step, t) - 0.9e-9
                    v = neuron(I_syn)
                    
                    self.simulation_log.update_neuron_traces(neuron_id, time_step, v*1e3, I_syn)
                    

        else:
            print("ERROR: There is no neural network configured!")
            