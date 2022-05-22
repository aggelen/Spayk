#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:24:08 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt

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

class GENESIS_Synapse:
    def __init__(self, params, io_neuron_idx=None, external_inputs=None, exc_inh=None):
        """
        by Wilson & Bower, 1989 in GENESIS
        """
        if 'tauExc' in params.keys():
            self.tauExc = params['tauExc']
        else:
            self.tauExc = 10
            
        if 'tauInh' in params.keys():
            self.tauInh = params['tauInh']
        else:
            self.tauInh = 10
            
        self.dt = params['dt']
        
        self.input_neuron_idx, self.output_neuron_idx = io_neuron_idx
        if type(self.input_neuron_idx) == list:
            self.no_input_neurons = len(self.input_neuron_idx)
            self.W = np.random.rand(len(self.input_neuron_idx))
        else:
            self.no_input_neurons = 1
            self.W = np.random.rand()
         
        #FIXME!
        if external_inputs is not None:
            self.no_ext_input_neurons = len(external_inputs)
            self.external_inputs = external_inputs
            self.W_ext = np.random.rand(len(self.external_inputs))
            
            self.last_zs_ext = np.zeros(self.no_ext_input_neurons)
            self.last_spikes_ext = np.zeros(self.no_ext_input_neurons)
            self.last_gs_ext = np.zeros(self.no_ext_input_neurons)
            
            self.taus_ext = np.where(np.ones_like(self.external_inputs), self.tauExc, self.tauInh)
            self.fdts_ext = np.exp(-self.dt / self.taus_ext)
            self.channel_conductance_history_ext = []
        
        self.last_zs = np.zeros(self.no_input_neurons)
        self.last_spikes = np.zeros(self.no_input_neurons)
        self.last_gs = np.zeros(self.no_input_neurons)
                
        if exc_inh is not None:
            self.exc_inh = exc_inh
        else:
            self.exc_inh = np.ones_like(self.input_neuron_idx)
        
        self.taus = np.where(exc_inh, self.tauExc, self.tauInh)
        self.fdts = np.exp(-self.dt / self.taus)
        self.channel_conductance_history = []
        
    def set_W(self, W):
        self.W = W
                    
    def g_syns(self, spikes, t, external_data=None):
        zs = (self.last_zs * self.fdts) + (self.taus * (1-self.fdts)/self.dt)*self.last_spikes
        gs = (self.last_gs * self.fdts) + (self.last_zs * self.taus * (1-self.fdts))
        self.last_zs = zs
        self.last_gs = gs
        self.last_spikes = spikes[self.input_neuron_idx]
        
        if external_data is not None:
            zs_ext = (self.last_zs_ext * self.fdts_ext) + (self.taus_ext * (1-self.fdts_ext)/self.dt)*self.last_spikes_ext
            gs_ext = (self.last_gs_ext * self.fdts_ext) + (self.last_zs_ext * self.taus_ext * (1-self.fdts_ext))
            self.last_zs_ext = zs_ext
            self.last_gs_ext = gs_ext
            self.last_spikes_ext = external_data
            
            # gs = np.append(gs,gs_ext)
        else:
            gs_ext = None
        
        return gs, gs_ext
    
    def I(self, vs, spikes, t, connection_spikes=None):
        # n0 = self.input_neurons[0]
        # self.g_syn(spike=tissue.log_spikes[0][tid],t=t)
        
        #FIXME!
        I = 0.0
        
        if connection_spikes is not None:
            external_data = connection_spikes[self.external_inputs]
        else:
            external_data = None
        
        gsyns, gsyn_ext = self.g_syns(spikes, t, external_data)
        self.channel_conductance_history.append(gsyns)
        
        Is = gsyns*-self.exc_inh*2e-3*vs[self.output_neuron_idx]*self.W
        
        if gsyn_ext is not None:        #all exc.
            Is_ext = -gsyn_ext*2e-3*vs[self.output_neuron_idx]*self.W_ext
            return sum(np.append(Is, Is_ext))
        else:
            return sum(Is)
        

        
       
        
        # for neuron_id in self.input_neuron_idx:
        #     # neuron_id = self.input_neuron_idx[0]
        #     spike = spikes[neuron_id]
        #     is_exc = self.exc_inh[neuron_id]
        #     gsyn = self.g_syn(spike,t,is_exc)*1e-4*vs[self.output_neuron_idx]*self.W[neuron_id]
        #     if is_exc:     #1 for exc, 0 for inh.
        #         I += -gsyn 
        #     else:
        #         I += gsyn 
        # return I
        
    # def g_syn(self, spike, t, exc=True):
    #     if exc:
    #         tau = self.tauExc
    #     else:
    #         tau = self.tauInh
            
    #     fdt = np.exp(-self.dt / tau)
    #     z = (self.last_z * fdt) + (tau * (1-fdt)/ self.dt)*self.last_spike
    #     g = (self.last_g * fdt) + (self.last_z * tau * (1-fdt))
    #     self.last_z = z
    #     self.last_g = g
    #     self.last_spike = spike
    #     return g
    
    # def I(self, vs, spikes, t):
    #     # n0 = self.input_neurons[0]
    #     # self.g_syn(spike=tissue.log_spikes[0][tid],t=t)
        
    #     #FIXME!
    #     I = 0.0
    #     for neuron_id in self.input_neuron_idx:
    #         # neuron_id = self.input_neuron_idx[0]
    #         spike = spikes[neuron_id]
    #         is_exc = self.exc_inh[neuron_id]
    #         gsyn = self.g_syn(spike,t,is_exc)*1e-4*vs[self.output_neuron_idx]*self.W[neuron_id]
    #         if is_exc:     #1 for exc, 0 for inh.
    #             I += -gsyn 
    #         else:
    #             I += gsyn 
    #     return I
    
    def plot_channel_conductances(self, ch_id, dt):
        g_hist = np.array(self.channel_conductance_history).T
        time = np.arange(g_hist.shape[1])*dt
        plt.figure()
        plt.plot(g_hist[ch_id])
        plt.xlabel('Time (ms)')
        plt.ylabel('uS')
        plt.title('Conductance of Channel between Neuron#{}'.format(ch_id))
        plt.grid()