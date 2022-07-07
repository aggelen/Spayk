#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:17:14 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt
from spayk.Utils import izhikevich_dynamics_selector

#%% Base Classes
class Neuron:
    def __init__(self):
        pass
    
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

# Models    
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
        
        if self.behaviour == 'synaptic' or self.behaviour == 'recurrent':
            if 'no_syn' in params.keys():
                self.no_syn = params['no_syn']
            else:
                raise Exception('Synaptic behaviour needs no_syn parameter. (# snyaptic conns)')
            self.syn_tau = params['syn_tau'] if 'syn_tau' in params.keys() else 10.0
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
        
    def autoconnect(self):
        self.w = np.random.uniform(0.245, 0.865, (self.no_neurons, self.no_syn)) / self.no_syn
    
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
        self.g  = np.where(current_spikes,
                           self.g + np.ones_like(self.g),
                           self.g - np.multiply(self.dt, self.g/self.syn_tau))
        
        self.g_history.append(self.g)
        
        I_syn = np.einsum('nm,m->n', self.w, np.multiply(self.g, self.E)) - np.multiply(np.einsum('nm,m->n', self.w, self.g), self.v)
        return I_syn
    
    def recurrent_syn_current(self, current_spikes, fired):
        I_syn = self.synaptic_current(current_spikes)
        
        self.g_rec = np.where(fired,
                          self.g_rec + np.ones_like(self.g_rec),
                          self.g_rec - np.multiply(self.dt, self.g_rec/self.syn_tau))
        
        I_rec = np.einsum('ij,j->i', self.w_rec, np.multiply(self.g_rec, self.E_rec - self.v))
        
        return I_syn + I_rec
    
    def forward(self, current_spikes=None):        
        fired = np.greater_equal(self.v, np.full(self.no_neurons, self.v_threshold))
        v = np.where(fired, self.C, self.v)
        u = np.where(fired, np.add(self.u, self.D), self.u)
        
        if self.behaviour == 'basic':
            I = self.I_inj
        elif self.behaviour == 'synaptic':
            I = self.synaptic_current(current_spikes)
        elif self.behaviour == 'recurrent':
            I = self.recurrent_syn_current(current_spikes, fired)
        
        diff_eqn = np.square(v)*0.04 + v*5.0 + np.full(self.no_neurons, 140.0) + I - self.u
        dv = np.where(fired, np.zeros_like(self.v), diff_eqn)
        du = np.where(fired, np.zeros_like(self.v), np.multiply(self.A, (np.multiply(self.B, v) - u)))

        self.v = np.minimum(np.full(self.no_neurons, self.v_threshold), v + np.multiply(dv, self.dt))
        self.u = u + np.multiply(du, self.dt)
        
        return self.v, self.u
    
class SRMLIFNeuron(NeuronGroup):
    def __init__(self, params):
        super().__init__()
        self.configure(params)
        self.current_spikes = None
        
        self.count = 0
        self.w_mean = []
        
        if params is None:
            #default
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
        self.spike_idx = self.n_syn - 1
        
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
        
        self.spike_idx = np.mod(self.spike_idx + 1, self.effective_time_window)

        idx = np.full(self.n_syn, 1, dtype=np.int32) * self.spike_idx
        coords = np.stack([idx, np.arange(self.n_syn)], axis=1)
        
        new_spikes = np.where(current_spikes,
                              np.full(self.n_syn, 0.0),
                              np.full(self.n_syn, 100000.0))
        
        self.t_tj[coords[:,0],coords[:,1]] = new_spikes
        
        # if self.supervise_status:
        #     if self.supervision_stimuli[self.count]:
        #         self.LTP()
        #     self.count += 1
            
        
        if self.t_rest > 0.0:
            self.v = self.rest()
        else:
            self.v = self.integrate()

        self.w_mean.append(self.w.mean())
        return self.v

    def rest(self):
        if self.stdp_status:
            if self.t_ti < self.tau_minus*7:
                self.LTD()
            
        self.t_rest = np.maximum(self.t_rest - self.dt, 0.0)
        neg_t_ti = -self.t_ti
        return self.v_th * (self.K1*np.exp(neg_t_ti/self.tau_m) - self.K2*(np.exp(neg_t_ti/self.tau_m) - np.exp(neg_t_ti/self.tau_s)))
    
    def integrate(self):
        # presyn_spike_time -> tj, postsyn_spike_time -> ti
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
        return self.v_th * (self.K1*np.exp(neg_t_ti/self.tau_m) - self.K2*(np.exp(neg_t_ti/self.tau_m) - np.exp(neg_t_ti/self.tau_s)))
    
    def LTD(self):
        ltd = np.where(np.logical_and(self.current_spikes, np.logical_not(self.presyn_spikes)),
                       np.full(self.n_syn, self.a_minus) * np.exp(-(self.t_ti/self.tau_minus)),
                       np.full(self.n_syn, 0.0))
        
        new_w = np.subtract(self.w, ltd)
        
        self.presyn_spikes = self.presyn_spikes | self.current_spikes
        self.w = np.clip(new_w, 0.0, 1.0)
    
    def LTP(self):
        last_spikes = np.min(self.t_tj, axis=0)

        ltp = np.where(last_spikes < self.t_ti,
                   np.full(self.n_syn, self.a_plus) * np.exp(-(last_spikes/self.tau_plus)),
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

# class IzhikevichNeuronGroup(NeuronGroup):
#     neuron_dynamics = {'regular_spiking': 0,
#                        'intrinsically_bursting': 1,
#                        'chattering': 2,
#                        'fast_spiking': 3,
#                        'thalamo_cortical': 4,
#                        'resonator': 5,
#                        'low_threshold_spiking': 6}
    
#     def __init__(self, params):
#         super().__init__()
#         self.no_neurons = params['no_neurons']
#         self.dynamics = params['dynamics']
        
#         self.set_dynamics(params)
        
#         self.v_threshold = 35.0
#         self.v_rest = -70.0
        
#         self.v = np.full(self.no_neurons, self.v_rest)
#         self.u = self.B*self.C
#         self.dt = params['dt']
        
#         self.I = np.zeros(self.no_neurons)
    
#     def inject_current(self, I):
#         self.I = I
    
#     def get_inputs(self, has_fired, v_reset):
#         return self.I
        
#     def forward(self):
#         has_fired = np.greater_equal(self.v, np.full(self.no_neurons, self.v_threshold))
#         v_reset = np.where(has_fired, self.C, self.v)
#         u_reset = np.where(has_fired, np.add(self.u, self.D), self.u)
        
#         I = self.get_inputs(has_fired, v_reset)
        
#         diff_eqn = np.square(v_reset)*0.04 + v_reset*5.0 + np.full(self.no_neurons, 140.0) + I - self.u
        
#         dv = np.where(has_fired, np.zeros(self.v.shape), diff_eqn)
        
#         du = np.where(has_fired,
#                       np.zeros(self.no_neurons),
#                       np.multiply(self.A, np.subtract(np.multiply(self.B, v_reset), u_reset)))

         
#         self.v = np.minimum(np.full(self.no_neurons, self.v_threshold),
#                        np.add(v_reset, np.multiply(dv, self.dt)))
        
#         self.u = np.add(u_reset, np.multiply(du, self.dt))
        
#         return self.v, self.u
    
#     def set_dynamics(self, params):
#         a, b, c, d = self.dynamics_selector(self.dynamics)
        
#         # Scale of the membrane recovery (lower values lead to slow recovery)
#         if not 'A' in params.keys():
#             self.A = np.full((self.no_neurons), a, dtype=np.float32)
#         else:
#             self.A = params['A']
#         # Sensitivity of recovery towards membrane potential (higher values lead to higher firing rate)
#         if not 'B' in params.keys():
#             self.B = np.full((self.no_neurons), b, dtype=np.float32)
#         else:
#             self.B = params['B']
#         # Membrane voltage reset value
#         if not 'C' in params.keys():
#             self.C = np.full((self.no_neurons), c, dtype=np.float32)
#         else:
#             self.C = params['C']
#         # Membrane recovery 'boost' after a spike
#         if not 'D' in params.keys():
#             self.D = np.full((self.no_neurons), d, dtype=np.float32)
#         else:
#             self.D = params['D']
        
#     def dynamics_selector(self, mode):
#         if mode == 'regular_spiking':
#             a = 0.02
#             b = 0.25
#             c = -65
#             d = 8
#             return a,b,c,d
#         elif mode == 'intrinsically_bursting':
#             a = 0.02
#             b = 0.2
#             c = -55
#             d = 4
#             return a,b,c,d
#         elif mode == 'chattering':
#             a = 0.02
#             b = 0.2
#             c = -50
#             d = 2
#             return a,b,c,d
#         elif mode == 'fast_spiking':
#             a = 0.1
#             b = 0.2
#             c = -65
#             d = 2
#             return a,b,c,d
#         elif mode == 'thalamo_cortical':
#             a = 0.02
#             b = 0.25
#             c = -65
#             d = 0.05
#             return a,b,c,d
#         elif mode == 'resonator':
#             a = 0.1
#             b = 0.25
#             c = -65
#             d = 8
#             return a,b,c,d
#         elif mode == 'low_threshold_spiking':
#             a = 0.02
#             b = 0.2
#             c = -65
#             d = 2
#             return a,b,c,d
#         elif mode == 'mixed':
#             return None,None,None,None
#         else:
#             raise Exception('Invalid Dynamics for Izhikevich Model')   
            
# class SynapticIzhikevichNeuronGroup(IzhikevichNeuronGroup):
#     def __init__(self, params):
#         self.no_connected_neurons = params['no_connected_neurons']
#         self.tau = params['tau']
        
#         if not 'W_in' in params.keys():
#             self.W_in = np.full((params['no_neurons'], self.no_connected_neurons), 0.5, dtype=np.float32)
#         else:
#             self.W_in = params['W_in']
            
#         if 'training_mode' in params.keys():
#             self.training_mode = params['training_mode']
#             self.desired_output_neurons, self.desired_output_currents = params['desired_output_currents']
#             self.current_counter = 0
#         else:
#             self.training_mode = False
        
#         # The reason this one is different is to allow broadcasting when subtracting v
#         self.E_in = np.zeros((self.no_connected_neurons), dtype=np.float32)
#         self.g_in = np.zeros((self.no_connected_neurons), dtype=np.float32)

#         self.syn_has_spiked = np.zeros((self.no_connected_neurons), dtype=bool)
        
#         # self.synaptic_scale = 1.0 / (20.0*self.no_connected_neurons)
#         # self.synaptic_scale = 1.0 / self.no_connected_neurons
#         self.synaptic_scale = 1.0
        
#         super(SynapticIzhikevichNeuronGroup, self).__init__(params)
        
#         self.g_history = []
        
#     def get_inputs_cuba(self, has_fired, v_reset):
#            # self.W_in, np.multiply(self.g_in, self.E_in)
#            I = np.sum(np.multiply(self.W_in, self.syn_has_spiked))*12.0
#            self.I = I
#            return self.I
           
#     def get_inputs(self, has_fired, v_reset):
#         # First, update synaptic conductance dynamics:
#         # - increment by one the current factor of synapses that fired
#         # - decrease by tau the conductance dynamics in any case
#         self.g_in  = np.where(self.syn_has_spiked,
#                                self.g_in + np.ones_like(self.g_in) * self.synaptic_scale,
#                               # self.g_in + np.ones_like(self.g_in)*23e-4,
#                               np.subtract(self.g_in, np.multiply(self.dt, np.divide(self.g_in, self.tau))))
        
#         self.g_history.append(self.g_in)
        
#         # We can now evaluate the synaptic input currents
#         # Isyn = Σ w_in(j)g_in(j)E_in(j) - (Σ w_in(j)g_in(j)).v(t)
#         I = np.subtract(np.einsum('nm,m->n', self.W_in, np.multiply(self.g_in, self.E_in)),
#                              np.multiply(np.einsum('nm,m->n', self.W_in, self.g_in), v_reset))
        
#         # self.I =np.where(I<0.0, 0.0, I)
        
#         if self.training_mode:
#             I[self.desired_output_neurons] = self.desired_output_currents[self.current_counter]
#             self.current_counter += 1
        
#         self.I = I

#         return self.I
    
#     def inject_spike_train(self, spike_train):
#         self.syn_has_spiked = spike_train
        
# class RecurrentIzhikevichNeuronGroup(SynapticIzhikevichNeuronGroup):
#     def __init__(self, params):
#         self.W = params['W']
#         self.E = params['E']
        
#         self.g = np.zeros(params['no_neurons'])

        
#         super(RecurrentIzhikevichNeuronGroup, self).__init__(params)
           
#     def get_inputs(self, has_fired, v_reset):
#         # First, update recurrent conductance dynamics:
#         # - increment by one the current factor of synapses that fired
#         # - decrease by tau the conductance dynamics in any case
#         self.g = np.where(has_fired,
#                           np.add(self.g, np.ones_like(self.g)*self.synaptic_scale),
#                           np.subtract(self.g, np.multiply(self.dt, np.divide(self.g, self.tau))))
       
#         # We can now evaluate the recurrent conductance
#         # I_rec = Σ wjgj(Ej -v(t))
#         i_rec = np.einsum('ij,j->i', self.W, np.multiply(self.g, np.subtract(self.E, v_reset)))

#         # Get the synaptic input currents from parent
#         i_in = super(RecurrentIzhikevichNeuronGroup, self).get_inputs(has_fired, v_reset)
        
#         # The actual current is the sum of both currents
#         i_op = i_in + i_rec

#         # Store a reference to this operation for easier retrieval
#         self.I = i_op
        
#         return i_op
    
# #%%
# class STDPIzhikevichNeuronGroup(SynapticIzhikevichNeuronGroup):
#     def __init__(self, params):
#         super(STDPIzhikevichNeuronGroup, self).__init__(params)
#         self.a_plus = params['a_plus']
#         self.tau_plus = params['tau_plus']
#         self.a_minus = params['a_minus']
#         self.tau_minus = params['tau_minus']
        
#         # The incoming spike times memory window
#         self.max_spikes = 700
#         self.t_spikes = np.full([self.max_spikes, params['no_connected_neurons']], 100000.0)
        
#         self.last_spike = 1000.0
        
#         self.new_spikes = np.full(params['no_connected_neurons'], False)

#         # The last spike time insertion index
#         self.t_spikes_idx = params['no_connected_neurons'] - 1
        
#         self.time = 0.0
        
#     def update_spikes_times(self):
#         # Increase the age of all the existing spikes by dt
#         self.t_spikes += np.ones_like(self.t_spikes) * self.dt
        
#         # new_spike_times = np.where(self.syn_has_spiked,
#         #                             np.full(self.no_connected_neurons, 0.0),
#         #                             np.full(self.no_connected_neurons, 100000.0))
        
#         # self.t_spikes = np.r_[new_spike_times.reshape(1,-1), self.t_spikes][:self.max_spikes, :]
        
        
        
#         # Increment last spike index (modulo max_spikes)
#         self.t_spikes_idx = np.mod(self.t_spikes_idx + 1, self.max_spikes)

#         # Create a list of coordinates to insert the new spikes
#         idx = np.full(self.no_connected_neurons, 1, dtype=np.int32) * self.t_spikes_idx
#         coords = np.stack([idx, np.arange(self.no_connected_neurons)], axis=1)

#         # Create a vector of new spike times (non-spikes are assigned a very high time)
#         new_spikes = np.where(self.syn_has_spiked,
#                               np.full(self.no_connected_neurons, 0.0),
#                               np.full(self.no_connected_neurons, 100000.))
        
#         self.t_spikes[coords[:,0],coords[:,1]] = new_spikes
                
        
        
#     def LTP(self):
#         # We only consider the last spike of each synapse from our memory
#         last_spikes = np.min(self.t_spikes, axis=0)

#         # Reward all last synapse spikes that happened after the previous neutron spike
#         # rewards = np.where(last_spikes < self.last_spike,
#         #                    np.full(self.no_connected_neurons, self.a_plus)*np.exp(-(last_spikes/self.tau_plus)),
#         #                    np.full(self.no_connected_neurons, 0.0))
        
#         rewards = np.full(self.no_connected_neurons, self.a_plus)*np.exp(-(last_spikes/self.tau_plus))
        
#         # Evaluate new weights
#         new_w = self.W_in + rewards
        
#         # Update with new weights clamped to [0,1]
#         # self.W_in = np.clip(new_w, 0.0, 1.0)
        
#         self.W_in = new_w
    
#     # Long Term synaptic Depression
#     def LTD(self):

#         # Inflict penalties on new spikes on synapses that have not spiked
#         # The penalty is equal for all new spikes, and inversely exponential
#         # to the time since the last spike
        
#         # if any(self.syn_has_spiked):
#         #     aykut = 5
        
#         penalties = np.where(np.logical_and(self.syn_has_spiked, np.logical_not(self.new_spikes)),
#                              np.full(self.no_connected_neurons, self.a_minus) * np.exp(-(self.last_spike/self.tau_minus)),
#                              np.full(self.no_connected_neurons, 0.0))
        
#         # Evaluate new weights
#         new_w = self.W_in - penalties
        
#         # Update the list of synapses that have spiked
#         self.new_spikes = self.syn_has_spiked | self.new_spikes
        
#         # Update with new weights clamped to [0,1]
#         # self.W_in = np.clip(new_w, 0.0, 1.0)
        
#         self.W_in = new_w
        
#     def firing_w_update(self):
#         # Reset the list of synapses that have spiked
#         # self.syn_has_spiked = np.full(self.no_connected_neurons, False).astype(bool)
#         self.new_spikes = np.full(self.no_connected_neurons, False).astype(bool)
#         self.LTP()

#     def standart_w_update(self):
#         # Apply long-term synaptic depression if we are still close to the last spike
#         # Note that if we unconditionally applied the LTD, the weights will slowly
#         # decrease to zero if no spike occurs.
#         if self.last_spike < self.tau_minus*7:
#             self.LTD()
#         else:
#             pass
        
#     def inject_spike_train(self, spike_train):
#         self.syn_has_spiked = spike_train
        
#     def forward(self):
#         self.update_spikes_times()
#         self.time += self.dt
#         self.last_spike += self.dt
        
#         has_fired = np.greater_equal(self.v, np.full(self.no_neurons, self.v_threshold))
#         v_reset = np.where(has_fired, self.C, self.v)
#         u_reset = np.where(has_fired, np.add(self.u, self.D), self.u)
        
#         # STDP
#         if any(has_fired):
#             # self.new_spikes = np.full(self.no_connected_neurons, False, dtype=bool)
#             self.last_spike = 0.0
#             self.firing_w_update()
#         else:
#             self.standart_w_update()
            
                
#         # I = self.get_inputs(has_fired, v_reset)
#         I = self.get_inputs(has_fired, v_reset)
        
#         diff_eqn = np.square(v_reset)*0.04 + v_reset*5.0 + np.full(self.no_neurons, 140.0) + I - self.u
        
#         dv = np.where(has_fired, np.zeros(self.v.shape), diff_eqn)
        
#         du = np.where(has_fired,
#                       np.zeros(self.no_neurons),
#                       np.multiply(self.A, np.subtract(np.multiply(self.B, v_reset), u_reset)))

         
#         self.v = np.minimum(np.full(self.no_neurons, self.v_threshold),
#                        np.add(v_reset, np.multiply(dv, self.dt)))
        
#         self.u = np.add(u_reset, np.multiply(du, self.dt))
        
#         return self.v, self.u
    
# #%%
# class STDPRecurrentIzhikevichNeuronGroup(STDPIzhikevichNeuronGroup):
#     def __init__(self, params):
#         super(STDPRecurrentIzhikevichNeuronGroup, self).__init__(params)
#         self.W = params['W']
#         self.E = params['E']
        
#         self.g = np.zeros(params['no_neurons'])

        
           
#     def get_inputs(self, has_fired, v_reset):
#         # First, update recurrent conductance dynamics:
#         # - increment by one the current factor of synapses that fired
#         # - decrease by tau the conductance dynamics in any case
#         self.g = np.where(has_fired,
#                           np.add(self.g, np.ones_like(self.g)*self.synaptic_scale),
#                           np.subtract(self.g, np.multiply(self.dt, np.divide(self.g, self.tau))))
       
#         # We can now evaluate the recurrent conductance
#         # I_rec = Σ wjgj(Ej -v(t))
#         i_rec = np.einsum('ij,j->i', self.W, np.multiply(self.g, np.subtract(self.E, v_reset)))

#         # Get the synaptic input currents from parent
#         i_in = super(STDPRecurrentIzhikevichNeuronGroup, self).get_inputs(has_fired, v_reset)
        
#         # The actual current is the sum of both currents
#         i_op = i_in + i_rec

#         # Store a reference to this operation for easier retrieval
#         self.I = i_op
        
#         return i_op
    
    
# #%%%% LIF
# class SpikeResponseLIF(NeuronGroup):
#     def __init__(self, params):
#         super().__init__()
        
#         self.n_syn, self.w, self.v_rest = params['n_syn'], params['w'], params['v_rest']
#         self.tau_rest, self.tau_m, self.tau_s = params['tau_rest'], params['tau_m'], params['tau_s']
#         self.K, self.K1, self.K2 = params['K'], params['K1'], params['K2']
        
#         self.dt = params['dt']
#         self.v = self.v_rest
#         self.t_rest = 0.0
        
#         if 'v_th' in params.keys():
#             self.v_th = params['v_th']
#         else:
#             self.v_th = self.n_syn / 4
        
#         if 'stdp_window' in params.keys():
#             self.stdp_window = params['stdp_window']
#         else:
#             self.stdp_window = 70
            
            
#         self.current_spikes = np.full(self.n_syn, False)

#         #self dt
#         self.spike_times = np.full((self.stdp_window, self.n_syn), 100000.0)

#         self.t_spikes_idx = self.n_syn - 1

#         self.time_since_last_spike = 1000.0
        
#     def forward(self, current_spikes):
#         self.current_spikes = current_spikes
#         self.time_update(current_spikes)
        
#         if self.t_rest:
#             v = self.rest()
#         else:
#             v = self.integrate()
        
#         self.v = v
        
#         return self.v
                
#     def time_update(self, current_spikes):
#         self.time_since_last_spike += self.dt
#         self.spike_times += self.dt
        
#         self.t_spikes_idx = np.mod(self.t_spikes_idx + 1, self.stdp_window)

#         idx = np.full(self.n_syn, 1, dtype=np.int32) * self.t_spikes_idx
#         coords = np.stack([idx, np.arange(self.n_syn)], axis=1)
        
#         new_spikes = np.where(current_spikes,
#                               np.full(self.n_syn, 0.0),
#                               np.full(self.n_syn, 100000.0))
        
#         self.spike_times[coords[:,0],coords[:,1]] = new_spikes
        
#     def rest(self):
#         self.resting_w_update()
#         self.t_rest = np.maximum(self.t_rest - self.dt, 0.0)
#         return self.update_v() 
    
     
#     def integrate(self):
#         new_states = self.update_v() + self.synaptic_EPSP()
        
#         if new_states > self.v_th:
#             return self.fire()
#         else:
#             return self.integration()
            
#     def integration_w_update(self):
#         pass
    
#     def firing_w_update(self):
#         pass
    
#     def resting_w_update(self):
#         pass
    
#     def integration(self):
#         self.integration_w_update()
#         return self.update_v() + self.synaptic_EPSP()
        
    
#     def fire(self):
#         self.firing_w_update()   
#         self.time_since_last_spike = 0.0
#         self.t_rest = self.tau_rest
        
#         return self.update_v()
        
#     def update_v(self):     #eta op
#         t = self.time_since_last_spike
#         psp = self.v_th * (self.K1*np.exp(-t/self.tau_m) - self.K2*(np.exp(-t/self.tau_m) - np.exp(-t/self.tau_s)))
#         return psp
    
#     def EPSP(self):
#         t = self.spike_times
#         return self.K *(np.exp(-t/self.tau_m) - np.exp(-t/self.tau_s))
    
#     def synaptic_EPSP(self):
#         EPSPs = np.where(np.logical_and(self.spike_times >=0, 
#                                               self.spike_times < self.time_since_last_spike - self.tau_rest),
#                                self.EPSP(),
#                                np.zeros_like(self.spike_times))
                          
#         return np.sum(self.w * EPSPs)
    
# class STDPSpikeResponseLIF(SpikeResponseLIF):
#     def __init__(self, params):
#         super().__init__(params)
#         self.a_plus, self.a_minus = params['stdp_params']['a_plus'], params['stdp_params']['a_minus']
#         self.tau_plus, self.tau_minus = params['stdp_params']['tau_plus'], params['stdp_params']['tau_minus']
        
#         self.syn_has_spiked = np.full(self.n_syn, False)
                
#     def LTD(self):
#         ltd = np.where(np.logical_and(self.current_spikes, np.logical_not(self.syn_has_spiked)),
#                        np.full(self.n_syn, self.a_minus) * np.exp(-(self.time_since_last_spike/self.tau_minus)),
#                        np.full(self.n_syn, 0.0))
        
#         new_w = np.subtract(self.w, ltd)
        
#         self.syn_has_spiked = self.syn_has_spiked | self.current_spikes
#         self.w = np.clip(new_w, 0.0, 1.0)
    
#     def LTP(self):
#         # We only consider the last spike of each synapse from our memory
#         last_spikes = np.min(self.spike_times, axis=0)

#         ltp = np.where(last_spikes < self.time_since_last_spike,
#                        np.full(self.n_syn, self.a_plus) * np.exp(-(last_spikes/self.tau_plus)),
#                        np.full(self.n_syn, 0.0))
        
#         new_w = self.w + ltp
#         self.w = np.clip(new_w, 0.0, 1.0)

#     def resting_w_update(self):
#         if self.time_since_last_spike < self.tau_minus*7:
#             self.LTD()
            
#     def integration_w_update(self):
#         if self.time_since_last_spike < self.tau_minus*7:
#             self.LTD()
    
#     def firing_w_update(self):
#         self.syn_has_spiked = np.full(self.n_syn, False)
#         self.LTP()