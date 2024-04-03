#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:24:08 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt

from spayk.Integrators import EulerIntegrator, RK4Integrator

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
    def __init__(self, params, io_neuron_idx=None, external_inputs=None, external_tissues=None, exc_inh=None):
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
        self.external_tissues = external_tissues

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
        
        self.stdp = STDP(A_plus=0.008, A_minus=0.008*1.10, tau_stdp=20)
        
        self.LTP_history = np.zeros((self.no_ext_input_neurons,2))
        # self.LTD_history = np.zeros((self.no_ext_input_neurons,2))
        self.gE_bar_update = np.ones((self.no_ext_input_neurons,1))

    def set_W(self, W):
        self.W = W
        
    def set_tissue(self, tissue):
        self.own_tissue = tissue

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
    
    def update_gebar_with_postspike(self):
        self.gE_bar_update[:, -1] = self.gE_bar_update[:, -1] + self.LTP_history[:,-1] * 0.024
        id_temp = self.gE_bar_update[:, -1] > 0.024
        self.gE_bar_update[id_temp, -1] = 0.024
    
    # def update_gebar(self,LTDs):
    #     # gE[it + 1] = np.copy(gE[it] - (dt / tau_syn_E) * gE[it] + (gE_bar_update[:, it] * presyn_spike_trains[:, it]).sum())
        
    #     self.gE_bar_update = np.c_[self.gE_bar_update, 
    #                                 np.copy(self.gE_bar_update[:, it] + LTDs*presyn_spike_trains[:, -1]*0.024)]
    
    #     id_temp = gE_bar_update[:, it + 1] < 0
    #     gE_bar_update[id_temp, it + 1] = 0.
    #     pass

    def I(self, i, vs, spikes, t, connection_spikes=None, synaptic_plasticity=False):
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
        
        
        if synaptic_plasticity == 'STDP' and self.external_tissues is not None and t > 0.1:
            # external_data  == presyn_spike_trains
            dP = -(self.dt / self.stdp.tau_stdp) * self.LTP_history[:, -2] + self.stdp.A_plus * external_data
            new_P = self.LTP_history[:,-1] + dP
            self.LTP_history = np.c_[self.LTP_history, new_P]
            
            LTD = self.own_tissue.LTD_history[i,-1]
            
            self.gE_bar_update = np.c_[self.gE_bar_update, 
                                        np.copy(self.gE_bar_update[:, -1] + LTD*external_data*0.024)]
        
            id_temp = self.gE_bar_update[:, -1] < 0
            self.gE_bar_update[id_temp, -1] = 0.
            

            
        if gsyn_ext is not None:        #all exc.
            Is_ext = -gsyn_ext*2e-3*vs[self.output_neuron_idx]*self.W_ext
            return sum(np.append(Is, Is_ext))
        else:
            return sum(Is)

    def plot_channel_conductances(self, ch_id, dt):
        g_hist = np.array(self.channel_conductance_history).T
        time = np.arange(g_hist.shape[1])*dt
        plt.figure()
        plt.plot(g_hist[ch_id])
        plt.xlabel('Time (ms)')
        plt.ylabel('uS')
        plt.title('Conductance of Channel between Neuron#{}'.format(ch_id))
        plt.grid()
        
#%% STDP

class STDP:
    def __init__(self, A_plus, A_minus, tau_stdp):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_stdp = tau_stdp
    
    def calculate_dW(self, time_diff):
        dW = np.zeros(len(time_diff))
        dW[time_diff <= 0] = self.A_plus * np.exp(time_diff[time_diff <= 0] / self.tau_stdp)
        dW[time_diff > 0] = -self.A_minus * np.exp(-time_diff[time_diff > 0] / self.tau_stdp)
        return dW 
    
    def calculate_LTP(self, presyn_spike_trains, dt):
      time_array = np.arange(presyn_spike_trains.shape[1])*dt

      # Initialize
      P = np.zeros(presyn_spike_trains.shape)
      for it in range(time_array.size - 1):
          dP = -(dt / self.tau_stdp) * P[:, it] + self.A_plus * presyn_spike_trains[:, it + 1]
          P[:, it + 1] = P[:, it] + dP
      return P
  
    def update_weights(self, presyn_spike_trains, dt):
        P = self.calculate_LTP(presyn_spike_trains, dt)
        time_array = np.arange(presyn_spike_trains.shape[1])*dt
        
        #LIF simulation
        V_th, V_reset = -55., -75.
        tau_m = 10.
        V_init, V_L = -65., -75.
        gE_bar, VE, tau_syn_E = 0.024, 0.0, 5.0
        gE_init = 0.024
        tref = 2.0
        
        tr = 0.
        v = np.zeros(time_array.size)
        v[0] = V_init
        M = np.zeros(time_array.size)
        gE = np.zeros(time_array.size)
        gE_bar_update = np.zeros(presyn_spike_trains.shape)
        gE_bar_update[:, 0] = gE_init  # note: gE_bar is the maximum value
        
        rec_spikes = [] 
        for it in range(time_array.size - 1):
            if tr > 0:
                v[it] = V_reset
                tr = tr - 1
            elif v[it] >= V_th:   # reset voltage and record spike event
                rec_spikes.append(it)
                v[it-1] =+ 30
                v[it] = V_reset
                M[it] = M[it] - self.A_minus
                gE_bar_update[:, it] = gE_bar_update[:, it] + P[:, it] * gE_bar
                id_temp = gE_bar_update[:, it] > gE_bar
                gE_bar_update[id_temp, it] = gE_bar
                tr = tref / dt            
            
            M[it + 1] = np.copy(M[it] - dt / self.tau_stdp * M[it])
            
            gE[it + 1] = np.copy(gE[it] - (dt / tau_syn_E) * gE[it] + (gE_bar_update[:, it] * presyn_spike_trains[:, it]).sum())
            gE_bar_update[:, it + 1] = np.copy(gE_bar_update[:, it] + M[it]*presyn_spike_trains[:, it]*gE_bar)
            id_temp = gE_bar_update[:, it + 1] < 0
            gE_bar_update[id_temp, it + 1] = 0.
        
            # calculate the increment of the membrane potential
            dv = (-(v[it] - V_L) - gE[it + 1] * (v[it] - VE)) * (dt / tau_m)
        
            # update membrane potential
            v[it + 1] = v[it] + dv
            
        rec_spikes = np.array(rec_spikes) * dt
        
        return v, rec_spikes, gE, P, M, gE_bar_update
    
#%% COBA Synapse
# I_syn = I_ext_AMPA + I_rec_AMPA + I_rec_NMDA + I_rec_GABA

class COBASynapse:
    def __init__(self, params):
        self.dt = params['dt']
        self.VE = params['VE']
        self.no_input_neurons = params['no_input_neurons']
        
        if 'g_ext_AMPA' in params.keys():
            self.integrator_ext_AMPA = RK4Integrator(dt=self.dt, f=self.d_s_AMPA)
            self.g_ext_AMPA = params['g_ext_AMPA']
            self.s_ext_AMPA = np.zeros(self.no_input_neurons)
            self.tau_AMPA = params['tau_AMPA']
            
        if 'g_rec_AMPA' in params.keys():
            self.integrator_rec_AMPA = RK4Integrator(dt=self.dt, f=self.d_s_AMPA)
            self.g_rec_AMPA = params['g_rec_AMPA']
            self.s_rec_AMPA = np.zeros(self.no_input_neurons)
            self.tau_AMPA = params['tau_AMPA']
            self.w_rec_AMPA = np.random.rand(1, self.no_input_neurons)
            
        if 'g_NMDA' in params.keys():
            self.g_NMDA = params['g_NMDA']
            self.Mg2 = params['Mg2+']
            self.alpha_NMDA = params['alpha_NMDA']
            self.tau_NMDA_rise = params['tau_NMDA_rise']
            self.tau_NMDA_decay = params['tau_NMDA_decay']
            
            self.x_NMDA = np.zeros(self.no_input_neurons)
            self.s_NMDA = np.zeros(self.no_input_neurons)
            
            self.integrator_x_NMDA = RK4Integrator(dt=self.dt, f=self.d_x_NMDA)
            self.integrator_s_NMDA = RK4Integrator(dt=self.dt, f=self.d_s_NMDA)
            
            self.w_NMDA = np.random.rand(1, self.no_input_neurons)
            
        if 'g_GABA' in params.keys():
            self.g_GABA = params['g_GABA']
            self.s_GABA = np.zeros(self.no_input_neurons)
            self.tau_GABA = params['tau_GABA']
            self.integrator_GABA = RK4Integrator(dt=self.dt, f=self.d_s_GABA)
            self.w_GABA = np.random.rand(1, self.no_input_neurons)
       
        
        self.spiked = False
    
    ###### AMPA
    def I_ext_AMPA(self, v, t):
        return self.g_ext_AMPA*(v-self.VE)*self.s_ext_AMPA
    
    def I_rec_AMPA(self, v, t):
        return self.g_rec_AMPA*(v-self.VE)*np.sum(self.w_rec_AMPA*self.s_rec_AMPA)
    
    def d_s_AMPA(self, t, s_ext_AMPA, extra_params):
        presyn_spikes = extra_params[0]
        ds = -s_ext_AMPA / self.tau_AMPA + np.where(self.spiked, presyn_spikes, np.zeros_like(s_ext_AMPA))
        return ds
    
    def update_s_ext_AMPA(self, t, presyn_spikes):
        #integrator(t,y0,extra_params as list)
        self.s_ext_AMPA = self.integrator_ext_AMPA(t, self.s_ext_AMPA, [presyn_spikes])
        return self.s_ext_AMPA
    
    def update_s_rec_AMPA(self, t, presyn_spikes):
        #integrator(t,y0,extra_params as list)
        self.s_rec_AMPA = self.integrator_rec_AMPA(t, self.s_rec_AMPA, [presyn_spikes])
        return self.s_rec_AMPA
    
    ###### NMDA 
    def I_NMDA(self, v, t):
        return (self.g_NMDA*(v-self.VE)*np.sum(self.w_NMDA*self.s_NMDA)) / (1 + self.Mg2*np.exp(-0.062*v/3.57))
    
    def d_s_NMDA(self, t, s_NMDA, extra_params):
        x = extra_params[0]
        ds = -s_NMDA / self.tau_NMDA_decay + self.alpha_NMDA*x*(np.ones_like(s_NMDA) - s_NMDA)
        return ds
    
    def d_x_NMDA(self, t, x_NMDA, extra_params):
        presyn_spikes = extra_params[0]
        dx= -x_NMDA / self.tau_NMDA_rise + np.where(self.spiked, presyn_spikes, np.zeros_like(x_NMDA))
        return dx
    
    def update_s_NMDA(self, t, presyn_spikes):
        #integrator(t,y0,extra_params as list)
        self.x_NMDA = self.integrator_x_NMDA(t, self.x_NMDA, [presyn_spikes])
        self.s_NMDA = self.integrator_s_NMDA(t, self.s_NMDA, [self.x_NMDA])
        return self.s_NMDA
    
    ###### GABA
    def I_GABA(self, v, t):
        return self.g_GABA*(v-self.VE)*np.sum(self.w_GABA*self.s_GABA)
    
    def d_s_GABA(self, t, s_GABA, extra_params):
        presyn_spikes = extra_params[0]
        ds = -s_GABA / self.tau_GABA + np.where(self.spiked, presyn_spikes, np.zeros_like(s_GABA))
        return ds
    
    def update_s_GABA(self, t, presyn_spikes):
        #integrator(t,y0,extra_params as list)
        self.s_GABA = self.integrator_GABA(t, self.s_GABA, [presyn_spikes])
        return self.s_GABA
    