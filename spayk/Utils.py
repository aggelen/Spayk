#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:27:52 2022

@author: aggelen
"""
import itertools
import networkx as nx
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from spayk.Synapses import VectorizedCOBASynapses

neuron_dynamics = {'regular_spiking': 0,
                   'intrinsically_bursting': 1,
                   'chattering': 2,
                   'fast_spiking': 3,
                   'thalamo_cortical': 4,
                   'resonator': 5,
                   'low_threshold_spiking': 6}

def dynamics_selector(mode):
    if mode == 'regular_spiking':
        a = 0.02
        b = 0.2
        c = -65
        d = 8
        return a,b,c,d
    elif mode == 'intrinsically_bursting':
        a = 0.02
        b = 0.2
        c = -55
        d = 4
        return a,b,c,d
    elif mode == 'chattering':
        a = 0.02
        b = 0.2
        c = -50
        d = 2
        return a,b,c,d
    elif mode == 'fast_spiking':
        a = 0.1
        b = 0.2
        c = -65
        d = 2
        return a,b,c,d
    elif mode == 'thalamo_cortical':
        a = 0.02
        b = 0.25
        c = -65
        d = 0.05
        return a,b,c,d
    elif mode == 'resonator':
        a = 0.1
        b = 0.25
        c = -65
        d = 2
        return a,b,c,d
    elif mode == 'low_threshold_spiking':
        a = 0.02
        b = 0.25
        c = -65
        d = 2
        return a,b,c,d
    elif mode == 'mixed':
        return None,None,None,None
    else:
        raise Exception('Invalid Dynamics for Izhikevich Model')   

def izhikevich_dynamics_selector(s):
    c0, c1, c2, c3, c4, c5, c6 = s == 0, s == 1, s == 2, s == 3, s == 4, s == 5, s == 6
  # rs, ib, ch, fs, tc, rz, lth
    cond_list = [c3, c5]  
    a_list = [np.full(s.shape, 0.1), np.full(s.shape, 0.1)]
    A = np.select(cond_list, a_list, 0.02)
    
    cond_list = [c4, c5, c6]  
    b_list = [np.full(s.shape, 0.25), np.full(s.shape, 0.25), np.full(s.shape, 0.25)]
    B = np.select(cond_list, b_list, 0.2)
    
    cond_list = [c1, c2]  
    c_list = [np.full(s.shape, -55), np.full(s.shape, -50)]
    C = np.select(cond_list, c_list, -65)
    
    cond_list = [c0,c1,c4]  
    d_list = [np.full(s.shape, 8.0), np.full(s.shape, 4.0), np.full(s.shape, 0.05)]
    D = np.select(cond_list, d_list, 2.0)
    return A,B,C,D

class ConnectionManager:
    def __init__(self, neural_network, receptors):
        self.neural_network = neural_network
        self.receptors = receptors
        
        self.no_stimuli = self.neural_network.no_stimuli
        self.no_neurons = self.neural_network.no_neurons
        self.no_channels = len(receptors)
        
        self.recurrent_connection_matrix = np.zeros((self.no_channels, self.no_neurons, self.no_neurons))
        self.stimuli_connection_matrix = np.zeros((self.no_channels, self.no_neurons, self.no_stimuli))     # matrix rows: neurons, cols, stims
        
    def create_pairs(self, source, target):
        # return list(itertools.product(source, target))
        return list(zip(source,target))
        
    def connect_neurons_to_stimuli(self, source_stim_idx, target_neuron_idx, channels, w=1.0):
        for channel in channels:
            channel_id = self.receptors.index(channel)
            pairs = self.create_pairs(target_neuron_idx, source_stim_idx)       # matrix rows: neurons, cols, stims
            for pid, p in enumerate(pairs):
                if isinstance(w, list):
                    self.stimuli_connection_matrix[channel_id][p] = w[pid]
                else:
                    self.stimuli_connection_matrix[channel_id][p] = w
    
    def connect_neurons_recurrent(self, pairs, channels, w):
        for channel in channels:
            channel_id = self.receptors.index(channel)
            for pid, p in enumerate(pairs):
                if isinstance(w, list):
                    self.recurrent_connection_matrix[channel_id][p] = w[pid]
                else:
                    self.recurrent_connection_matrix[channel_id][p] = w
    
    def render_synapses(self):
        self.synapses = VectorizedCOBASynapses(self.neural_network.dt,
                                               self.neural_network.neuron_configuration, 
                                               self.stimuli_connection_matrix, 
                                               self.recurrent_connection_matrix)
        
        self.synapses.render_parameter_matrices()
        
        return self.synapses
            
            
                    
# class ConnectionManager:
#     def __init__(self, neural_network):
#         self.neural_network = neural_network
#         #  TargetNeuronID; CHANNEL@SourceNeuronId; CHANNEL@SourceNeuronId; CHANNEL@SourceNeuronId; ....
#         self.source_list = []
#         self.channel_list = []
        
#         self.target_dict = defaultdict(list)
    
        
#         self.graph_string = """"""

#     def connect_one_to_one(self, target_neuron_idx, channels, source_neuron_idx):
#         ## from source neuron to target neuron         source ----> channel -----> target
#         for i in range(len(source_neuron_idx)):
#             current_line = self.target_dict[target_neuron_idx[i]]
#             if isinstance(channels, list):
#                 if channels[i] == 'EXT_AMPA':
#                     to_be_appended = channels[i]+"@E{}".format(source_neuron_idx[i])+"; "
#                 else:
#                     to_be_appended = channels[i]+"@N{}".format(source_neuron_idx[i])+"; "
#                 if not to_be_appended in current_line:
#                     current_line.append[to_be_appended] 
#             else:
#                 if channels == 'EXT_AMPA':
#                     to_be_appended = channels+"@E{}".format(source_neuron_idx[i])+"; "
#                 else:
#                     to_be_appended = channels+"@N{}".format(source_neuron_idx[i])+"; "
                
#                 if not to_be_appended in current_line:
#                     current_line.append(to_be_appended) 
                    
#     def connect_others(self, target_neuron_id, channels, other_neuron_idx):
#         for i in range(len(other_neuron_idx)):
#             current_line = self.target_dict[target_neuron_id]
#             pass
                       
#     def render_graph(self, model_save_path, model_name):
#         if model_save_path.endswith('.spayk'):
#             path = model_save_path
#         else:
#             path = model_save_path + ".spayk"
            
#         now_time = datetime.now()
#         formatted_time = now_time.strftime('%Y-%m-%d %H:%M:%S')    
            
#         with open(path, 'w') as f:
#             f.write('##### SPAYK NEURAL NETWORK GRAPH HEADER #####\n')
#             f.write('##### Model Name: {}\n'.format(model_name))
#             f.write('##### Creation Date: {}\n'.format(formatted_time))
#             f.write('\n')
#             f.write('##### SPAYK NEURAL NETWORK GRAPH NODES #####\n')
#             for n_id, n in enumerate(self.neural_network.neuron_list):
#                 f.write('N{}; {}\n'.format(n_id, n.type))
                
#             f.write('\n')
#             f.write('##### SPAYK NEURAL NETWORK GRAPH EDGES #####\n')
#             for k, v in self.target_dict.items():
#                 line = "N{}; ".format(k)
#                 for cnn in v:
#                     line += cnn
#                 line = line.rstrip()[:-1]
#                 line += "\n"
#                 f.write(line)
            
            
#     def load_graph(self, path):
#         with open(path) as file:
#             counter = -1
#             is_reading_edges = False
#             for line in file:
#                 counter += 1
#                 if counter == 1:
#                     model_name = line[18:]
#                 if counter == 2:
#                     creation_date = line[21:]
                
#                 if is_reading_edges:
#                     self.neural_network.add_connection(line)
                
#                 if line == '##### SPAYK NEURAL NETWORK GRAPH EDGES #####\n':
#                     is_reading_edges = True
                    
