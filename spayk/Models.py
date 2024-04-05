#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:17:14 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt
from spayk.Utils import izhikevich_dynamics_selector

# Ref.
# https://neuronaldynamics.epfl.ch/online/Ch3.S1.html (Synapses, channel models)
# [Masq2008] Spike Timing Dependent Plasticity Finds the Start of Repeating Patterns in Continuous Spike Trains (Spike respnse model, params)

#%% Base Classes
class Neuron:
    def __init__(self):
        pass
    
    def __call__(self, data=None):
        if data is None:
            return self.forward()
        else:
            return self.forward(data)

class NeuronGroup:
    def __init__(self):
        pass

    def __call__(self, data=None):
        if data is None:
            return self.forward()
        else:
            return self.forward(data)

    def prepare(self, dt):
        self.dt = dt

    def forward(self, data):
        pass

    def weight_update(self, params=None):
        pass


#%% LIF

class SynapticLIFNeuron(Neuron):
    def __init__(self, params):
        super().__init__()
        self.dt = params['dt']
        self.VL = params['VL']
        self.Vth = params['Vth']
        self.Vreset = params['Vreset']
        self.Cm = params['Cm']
        self.gL = params['gL']
        self.tau_ref = params['tau_ref']
                
        self.t_rest = 0.0
        self.v = self.VL
        self.spiked = False
        
        self.synapses = None
        self.logs = {'s_ext_AMPA': []}
        
    def calculate_synaptic_current(self, time_step, t):        
        I_syn = 0.0
        
        for s in self.synapses.channel_stack:
            if s == 'ext_AMPA':
                #FIXME : [0] -> multiple ext channels?   
                presyn_spikes = self.synapses.sources['ext_AMPA'][0].spikes[0,time_step]
                s_ext_AMPA = self.synapses.update_s_ext_AMPA(t, presyn_spikes)
                I_ext_AMPA = self.synapses.I_ext_AMPA(self.v, t)
                I_syn += I_ext_AMPA
                
                self.logs['s_ext_AMPA'].append(s_ext_AMPA[0])
                
        return I_syn
        
    def forward(self, I_syn):
        
        if self.t_rest > 0.0:
            #rest
            self.t_rest -= self.dt
            
        else:
            self.integrate(I_syn)
            
            if self.v >= self.Vth:
                self.fire()
        
        if self.spiked:
            self.spiked = False
            return self.v + 40e-3
        else:
            return self.v
        
    def integrate(self, I_syn):
        dv = (-self.gL*(self.v-self.VL) - I_syn)/self.Cm
        self.v += dv*self.dt
        
    def fire(self):
        self.v = self.Vreset
        self.t_rest = self.tau_ref
        self.spiked = True


#%% Models
class IzhikevichNeuronGroup(NeuronGroup):
    def __init__(self, params):
        super().__init__()
        self.no_neurons = params['no_neurons']

        self.behaviour = params['behaviour'] if 'behaviour' in params.keys() else 'basic'

        if 'behaviour' in params.keys():
            self.behaviour = params['behaviour']
        else:
            self.behaviour = 'basic'

        self.set_dynamics(params)

        self.v_threshold = 35.0
        self.v_rest = -65.0

        self.v = np.full(self.no_neurons, self.v_rest)
        self.u = self.B*self.C

        self.I_inj = 0
        
        self.output_spikes = None

        if self.behaviour == 'synaptic' or self.behaviour == 'recurrent':
            if 'no_syn' in params.keys():
                self.no_syn = params['no_syn']
            else:
                raise Exception('Synaptic behaviour needs no_syn parameter. (# snyaptic conns)')
                
            self.syn_tau = params['syn_tau'] if 'syn_tau' in params.keys() else 10.0
            if 'no_syn' in params.keys():
                self.E = params['E']
            else:
                self.E = np.zeros((self.no_syn), dtype=np.float32)
                
            self.g = np.zeros((self.no_syn), dtype=np.float32)
            # self.synaptic_scale = (1.0 / self.no_syn)*20    # if %20 input?
            self.g_history = []

        if self.behaviour == 'recurrent':
            self.g_rec = np.zeros((self.no_neurons), dtype=np.float32)
            self.E_rec = np.zeros((self.no_neurons), dtype=np.float32)

    def set_dynamics(self, params):
        #regular
        a, b, c, d = 0.02, 0.2, -65, 8

        if not 'A' in params.keys():
            self.A = np.full((self.no_neurons), a, dtype=np.float32)
        else:
            self.A = params['A']
        if not 'B' in params.keys():
            self.B = np.full((self.no_neurons), b, dtype=np.float32)
        else:
            self.B = params['B']
        if not 'C' in params.keys():
            self.C = np.full((self.no_neurons), c, dtype=np.float32)
        else:
            self.C = params['C']
        if not 'D' in params.keys():
            self.D = np.full((self.no_neurons), d, dtype=np.float32)
        else:
            self.D = params['D']

    def autoconnect(self, scale=1.0):
        self.w = np.random.uniform(0.3, 0.7, (self.no_neurons, self.no_syn)) / self.no_syn
        # self.w = np.random.uniform(0.4, 0.6, (self.no_neurons, self.no_syn)) / self.no_syn
        self.w *= scale

    def set_architecture(self, params):
        if 'dynamics' in params.keys():
            self.A, B, self.C, self.D = izhikevich_dynamics_selector(params['dynamics'])
            # pass
        if 'w' in params.keys():
            self.w = params['w']

        if 'E_rec' in params.keys():
            self.E_rec = params['E_rec']

        if 'w_rec' in params.keys():
            self.w_rec = params['w_rec']

    def inject_current(self, I):
        self.I_inj = I

    def synaptic_current(self, current_spikes):
        # exponential * heaviside trick -> http://www.kaizou.org/2018/07/simulating-spiking-neurons-with-tensorflow.html
        self.g  = np.where(current_spikes,
                           self.g + np.ones_like(self.g),
                           self.g - np.multiply(self.dt, self.g/self.syn_tau))

        self.g_history.append(self.g)
        
        # https://neuronaldynamics.epfl.ch/online/Ch3.S1.html
        # Isyn(t)= w*gsyn(t)*(v(t)−Esyn)
        
        I_syn = np.einsum('nm,m->n', self.w, np.multiply(self.g, self.E)) - np.multiply(np.einsum('nm,m->n', self.w, self.g), self.v)
        return I_syn

    def recurrent_syn_current(self, current_spikes):
        I_syn = self.synaptic_current(current_spikes)

        self.g_rec = np.where(np.greater_equal(self.v, self.v_threshold),
                              self.g_rec + np.ones_like(self.g_rec),
                              self.g_rec - np.multiply(self.dt, self.g_rec/self.syn_tau))

        I_rec = np.einsum('ij,j->i', self.w_rec, np.multiply(self.g_rec, self.E_rec - self.v))

        return I_syn + I_rec

    def forward(self, current_spikes=None):
        if self.behaviour == 'basic':
            I = self.I_inj
        elif self.behaviour == 'synaptic':
            I = self.synaptic_current(current_spikes)
        elif self.behaviour == 'recurrent':
            I = self.recurrent_syn_current(current_spikes)
        
        #izhikevich's implementation
        fired = np.greater_equal(self.v, self.v_threshold)
        
        self.output_spikes = fired
        
        v = np.where(fired, self.C, self.v)
        u = np.where(fired, self.u + self.D, self.u)
            
        dv = np.square(v)*0.04 + v*5.0 + np.full(self.no_neurons, 140.0) + I - u
        du = np.multiply(self.A, np.subtract(np.multiply(self.B, v), u))
            
        self.v = np.minimum(self.v_threshold, v + dv*self.dt)
        self.u = u + du*self.dt
        
        return self.v, self.u
    
#%% Spike Response Model
class SRMLIFNeuron(NeuronGroup):
    def __init__(self, params):
        super().__init__()
        self.configure(params)
        self.current_spikes = None

        self.count = 0
        self.w_mean = []
        
        self.output_spikes = None

        if params is None:
            pass
        else:
            pass

    def configure(self, params):
        self.n_syn = params['n_synapses']
        self.dt = params['dt']
        self.t_rest = 0.0

        self.effective_time_window = 70

        # presyn_spike_time -> tj, postsyn_spike_time -> ti
        self.t_ti = 1e3
        self.t_tj = np.full((self.effective_time_window, self.n_syn), 100000.0)
        # self.spike_idx = self.n_syn - 1

        # self.w = params['w'] if 'w' in params.keys() else np.full((self.n_syn), 0.475, dtype=np.float32)
        self.w = params['w'] if 'w' in params.keys() else np.random.uniform(0.125, 0.565, self.n_syn)

        self.v_rest = params['v_rest'] if 'v_rest' in params.keys() else 0.0
        self.v_th = params['v_th'] if 'v_th' in params.keys() else self.n_syn / 4
        self.v = self.v_rest

        self.tau_m = params['tau_m'] if 'tau_m' in params.keys() else 10.0
        self.tau_s = params['tau_s'] if 'tau_s' in params.keys() else 2.5
        self.tau_rest = params['tau_rest'] if 'tau_rest' in params.keys() else 1.0

        self.K = params['K'] if 'K' in params.keys() else 2.1
        self.K1 = params['K1'] if 'K1' in params.keys() else 2.0
        self.K2 = params['K2'] if 'K2' in params.keys() else 4.0

        self.stdp_status = params['stdp_on'] if 'stdp_on' in params.keys() else False
        self.supervise_status = params['supervise_on'] if 'supervise_on' in params.keys() else False
        self.supervision_stimuli = params['supervision_stimuli'] if 'supervision_stimuli' in params.keys() else None

        if self.stdp_status:
            self.a_plus, self.a_minus = params['stdp_params']['a_plus'], params['stdp_params']['a_minus']
            self.tau_plus, self.tau_minus = params['stdp_params']['tau_plus'], params['stdp_params']['tau_minus']
            self.presyn_spikes = np.full(self.n_syn, False)

    def autoconnect(self):
        pass

    def set_architecture(self):
        pass

    def forward(self, current_spikes):
        self.current_spikes = current_spikes

        # update time       
        self.t_ti += self.dt
        self.t_tj += self.dt

        new_spike_times = np.where(current_spikes,
                                   np.full(current_spikes.size, 0.0),
                                   np.full(current_spikes.size, 100000.0))
        
        self.t_tj = np.r_[[new_spike_times], self.t_tj]
        self.t_tj = np.delete(self.t_tj, -1, 0)

        if self.t_rest > 0.0:
            self.v = self.rest()
        else:
            self.v = self.integrate()

        self.w_mean.append(self.w.mean())
        self.output_spikes = (self.v > self.v_th).astype(int)
        return self.v

    def rest(self):
        if self.stdp_status:
            if self.t_ti < self.tau_minus*7:
                self.LTD()
                
        #sometimes goes neg!
        self.t_rest = np.maximum(self.t_rest - self.dt, 0.0)
        
        neg_t_ti = -self.t_ti
        # [Masq2008] page8, SRM equations, eta!
        return self.v_th * (self.K1*np.exp(neg_t_ti/self.tau_m) - self.K2*(np.exp(neg_t_ti/self.tau_m) - np.exp(neg_t_ti/self.tau_s)))

    def integrate(self):
        # presyn_spike_time -> tj, postsyn_spike_time -> ti
        # equations from [Masq2008] page8 epsilon and eta functions
        
        #Update memb. potential
        neg_t_ti, neg_t_tj = -self.t_ti, -self.t_tj
        spike_response = self.v_th * (self.K1*np.exp(neg_t_ti/self.tau_m) - self.K2*(np.exp(neg_t_ti/self.tau_m) - np.exp(neg_t_ti/self.tau_s)))
        EPSPs = self.K *(np.exp(neg_t_tj/self.tau_m) - np.exp(neg_t_tj/self.tau_s))

        heaviside_cond = np.logical_and(self.t_tj >=0, self.t_tj < self.t_ti - self.tau_rest)
        EPSPs_Heaviside = np.where(heaviside_cond, EPSPs, np.zeros_like(self.t_tj))

        v = spike_response + np.sum(self.w * EPSPs_Heaviside)

        # fire?
        if v > self.v_th:
            return self.fire()
        else:
            if self.stdp_status:
                if self.t_ti < self.tau_minus*7:
                    self.LTD()
            return v

    def fire(self):
        if self.stdp_status:
            self.presyn_spikes = np.full(self.n_syn, False)
            self.LTP()

        self.t_ti = 0.0
        self.t_rest = self.tau_rest
        neg_t_ti = -self.t_ti
        #eta function
        return self.v_th * (self.K1*np.exp(neg_t_ti/self.tau_m) - self.K2*(np.exp(neg_t_ti/self.tau_m) - np.exp(neg_t_ti/self.tau_s)))

    def LTD(self):
        ltd = np.where(np.logical_and(self.current_spikes, np.logical_not(self.presyn_spikes)),
                       np.full(self.n_syn, self.a_minus) * np.exp(-(self.t_ti/self.tau_minus)),
                       np.full(self.n_syn, 0.0))

        new_w = np.subtract(self.w, ltd)

        self.presyn_spikes = self.presyn_spikes | self.current_spikes
        self.w = np.clip(new_w, 0.0, 1.0)

    def LTP(self):
        tj = np.min(self.t_tj, axis=0)
        
        ltp = np.where(tj < self.t_ti,
                       np.full(self.n_syn, self.a_plus) * np.exp(-(tj/self.tau_plus)),
                       np.full(self.n_syn, 0.0))

        new_w = self.w + ltp
        self.w = np.clip(new_w, 0.0, 1.0)

class SRMLIFNeuronGroup(NeuronGroup):
    def __init__(self, group_params, neuron_params):
        super().__init__()
        self.no_neurons = group_params['no_neurons']
        self.neurons = []
        for i in range(self.no_neurons):
            if isinstance(neuron_params, dict):
                self.neurons.append(SRMLIFNeuron(neuron_params))
            elif isinstance(neuron_params, list):
                self.neurons.append(SRMLIFNeuron(neuron_params[i]))

    def forward(self, current_spikes):
        vs = []
        for neuron in self.neurons:
            v = neuron(current_spikes)
            vs.append(v)

        return vs
