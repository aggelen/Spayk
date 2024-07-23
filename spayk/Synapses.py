#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:24:08 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt

from spayk.Integrators import EulerIntegrator, RK4Integrator


#%% New Core

class SynapseGroup:
    def __init__(self, source=None, target=None, params=None, state_labels=None):
        self.params = params
        self.source = source
        self.target = target
        self.channels = []
        self.state_labels = []
        
    def AMPA(self, gs, ws, state_label):
        if isinstance(gs, float):
            self.g_AMPA = gs
        else:
            raise NotImplementedError()
            
        if isinstance(ws, np.ndarray):
            self.w_AMPA = ws
        elif ws is None:
                self.w_AMPA = None
        else:
            raise NotImplementedError()
        
        self.channels.append("AMPA"),
        self.state_labels.append(state_label)
        
    def AMPA_EXT(self, gs, ws, state_label):
        if isinstance(gs, float):
            self.g_AMPA_ext = gs
        else:
            raise NotImplementedError()
            
        if isinstance(ws, np.ndarray):
            self.w_AMPA_ext = ws
        elif ws is None:
                self.w_AMPA_ext = None
        else:
            raise NotImplementedError()
        
        self.channels.append("AMPA_EXT")
        self.state_labels.append(state_label)
        
    def NMDA(self, gs, ws, state_label):
        if isinstance(gs, float):
            self.g_NMDA = gs
        else:
            raise NotImplementedError()
            
        if isinstance(ws, np.ndarray):
            self.w_NMDA = ws
        elif ws is None:
            self.w_NMDA = None
        else:
            raise NotImplementedError()
        
        self.channels.append("NMDA")
        self.state_labels.append(state_label)
        
    def GABA(self, gs, ws, state_label):
         if isinstance(gs, float):
             self.g_GABA = gs
         else:
             raise NotImplementedError()
             
         if isinstance(ws, np.ndarray):
             self.w_GABA = ws
         elif ws is None:
             self.w_GABA = None
         else:
             raise NotImplementedError()
         
         self.channels.append("GABA")
         self.state_labels.append(state_label)


#%% OLD CORE


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
class ChannelParameters:
    experimental_pyramidal_cell_params = {"g_ext_ampa": 2.1,
                                        "g_rec_ampa": 0.05,
                                        "g_rec_nmda": 0.165,
                                        "g_rec_gaba": 1.3}

    experimental_interneuron_params = {"g_ext_ampa": 1.62,
                                       "g_rec_ampa": 0.04,
                                       "g_rec_nmda": 0.13,
                                       "g_rec_gaba": 1.0}


    AMPA_ext_to_pyramidal_cell = {'g_ext_ampa': 2.1,
                                  'VE': 0,
                                  'tau_AMPA': 2e-3}
    
    AMPA_ext_to_interneuron = {'g_ext_ampa': 1.62,
                               'VE': 0,
                               'tau_AMPA': 2e-3}
    
    NMDA = {'Mg2+': 1e-3,
            'alpha_NMDA': 0.5e3,
            'tau_NMDA_rise': 2e-3,
            'tau_NMDA_decay': 100e-3}
        
    @staticmethod
    def get_channel_matrices(neuron_conf, conn_matrix_stim, conn_matrix_rec):
        
        conn_ext_ampa = conn_matrix_stim[0]
        conn_rec_ampa = conn_matrix_rec[1]
        conn_rec_nmda = conn_matrix_rec[2]
        
        g_ext_ampa = []
        g_rec_ampa = []
        g_rec_nmda = []
        g_rec_gaba = []
        
        for neuron_type in neuron_conf:
            if neuron_type == 'PyramidalCell':
                g_ext_ampa.append(np.full(conn_ext_ampa.shape[1], ChannelParameters.experimental_pyramidal_cell_params['g_ext_ampa']))
                g_rec_ampa.append(np.full(conn_rec_ampa.shape[1], ChannelParameters.experimental_pyramidal_cell_params['g_rec_ampa']))
                g_rec_nmda.append(np.full(conn_rec_ampa.shape[1], ChannelParameters.experimental_pyramidal_cell_params['g_rec_nmda']))
                g_rec_gaba.append(np.full(conn_rec_ampa.shape[1], ChannelParameters.experimental_pyramidal_cell_params['g_rec_gaba']))
            elif neuron_type == 'Interneuron':
                g_ext_ampa.append(np.full(conn_ext_ampa.shape[1], ChannelParameters.experimental_interneuron_params['g_ext_ampa']))     
                g_rec_ampa.append(np.full(conn_rec_ampa.shape[1], ChannelParameters.experimental_interneuron_params['g_rec_ampa']))
                g_rec_nmda.append(np.full(conn_rec_ampa.shape[1], ChannelParameters.experimental_interneuron_params['g_rec_nmda']))
                g_rec_gaba.append(np.full(conn_rec_ampa.shape[1], ChannelParameters.experimental_interneuron_params['g_rec_gaba']))
                
        ext_ampa = {'g_ext_ampa': np.array(g_ext_ampa),
                    'tau_ext_ampa': 2e-3*np.ones_like(conn_matrix_stim[0]),
                    'VE': np.zeros_like(conn_matrix_stim[0])}
        
        rec_ampa = {'g_rec_ampa': np.array(g_rec_ampa),
                    'tau_rec_ampa': 2e-3*np.ones_like(conn_matrix_rec[1]),
                    'VE': np.zeros_like(conn_matrix_rec[1])}
        
        rec_nmda = {'VE': np.zeros_like(conn_matrix_rec[2]),
                    'Mg2+': 1e-3,
                    'alpha_NMDA': 0.5e-3,
                    'tau_NMDA_rise': 2e-3,
                    'tau_NMDA_decay': 100e-3,
                    'g_rec_nmda': np.array(g_rec_nmda)}
        
        rec_gaba = {'VE': 0,
                    'VL': -70e-3,
                    'g_rec_gaba': np.array(g_rec_gaba),
                    'tau_GABA': 5e-3}
        
        return ext_ampa, rec_ampa, rec_nmda, rec_gaba

    
class VectorizedCOBASynapses:
    def __init__(self, dt, neuron_configuration, stimuli_connections, recurrent_connections):
        self.dt = dt
        self.neuron_configuration = neuron_configuration
        self.stimuli_connections = stimuli_connections
        self.recurrent_connections = recurrent_connections
        
        self.integrator_ext_AMPA = RK4Integrator(dt=self.dt, f=self.d_s_AMPA)
        self.s_ext_AMPA = np.zeros_like(self.stimuli_connections[0])
        
        self.integrator_rec_AMPA = RK4Integrator(dt=self.dt, f=self.d_s_AMPA)
        self.s_rec_AMPA = np.zeros_like(self.recurrent_connections[1])
        
        self.x_NMDA = np.zeros_like(self.recurrent_connections[2])
        self.s_NMDA = np.zeros_like(self.recurrent_connections[2])
        
        self.integrator_x_NMDA = RK4Integrator(dt=self.dt, f=self.d_x_NMDA)
        self.integrator_s_NMDA = RK4Integrator(dt=self.dt, f=self.d_s_NMDA)
        
        self.s_GABA = np.zeros_like(self.recurrent_connections[3])
        self.integrator_GABA = RK4Integrator(dt=self.dt, f=self.d_s_GABA)
        
        self.s_ext_ampa_log = []
        
    def render_parameter_matrices(self):
        ext_ampa, rec_ampa , rec_nmda, rec_gaba = ChannelParameters.get_channel_matrices(self.neuron_configuration, self.stimuli_connections , self.recurrent_connections)
        
        self.parameter_matrices = {'ext_ampa': ext_ampa,
                                   'rec_ampa': rec_ampa,
                                   'rec_nmda': rec_nmda,
                                   'rec_gaba': rec_gaba}
    
    # AMPA
    def I_ext_AMPA(self, v, t):
        v_matrix = np.tile(np.expand_dims(v, 1), self.s_ext_AMPA.shape[1])
        return self.parameter_matrices['ext_ampa']['g_ext_ampa']*(v_matrix-self.parameter_matrices['ext_ampa']['VE'])*self.s_ext_AMPA*self.stimuli_connections[0]
    
    def update_s_ext_AMPA(self, t, presyn_spikes, postsyn_spikes):
        #integrator(t,y0,extra_params as list)
        self.s_ext_AMPA = self.integrator_ext_AMPA(t, self.s_ext_AMPA, [presyn_spikes, postsyn_spikes, 'ext_ampa'])
        
    def d_s_AMPA(self, t, s_ext_AMPA, extra_params):
        presyn_spikes, postsyn_spikes, mode = extra_params
        if mode == 'ext_ampa':
            # is_spiked = np.tile(np.expand_dims(postsyn_spikes, 1), presyn_spikes.shape[0])
            # ds = -s_ext_AMPA / self.parameter_matrices[mode]['tau_'+mode] + np.where(is_spiked, presyn_spikes, np.zeros_like(s_ext_AMPA))
            presyn_trains = np.tile(np.expand_dims(presyn_spikes, 0), (postsyn_spikes.shape[0],1))
            ds = (-s_ext_AMPA / self.parameter_matrices[mode]['tau_'+mode]) + presyn_trains
            
        else:
            postsyn_trains = np.tile(np.expand_dims(postsyn_spikes, 0), (postsyn_spikes.shape[0],1))
            if len(np.argwhere(postsyn_trains)):
                aykut = 4
            ds = (-s_ext_AMPA / self.parameter_matrices[mode]['tau_'+mode]) + postsyn_trains
        return ds
    
    # REC AMPA
    def update_s_rec_AMPA(self, t, presyn_spikes, postsyn_spikes):
        #integrator(t,y0,extra_params as list)
        self.s_rec_AMPA = self.integrator_rec_AMPA(t, self.s_rec_AMPA, [presyn_spikes, postsyn_spikes, 'rec_ampa'])
    
    def I_rec_AMPA(self, v, t):
        v_matrix = np.tile(np.expand_dims(v, 1), self.s_rec_AMPA.shape[1])
        return self.parameter_matrices['rec_ampa']['g_rec_ampa']*(v_matrix-self.parameter_matrices['rec_ampa']['VE'])*(self.recurrent_connections[1]*self.s_rec_AMPA)
         
    ###### NMDA 
    def update_s_NMDA(self, t, presyn_spikes, postsyn_spikes):
        #integrator(t,y0,extra_params as list)
        self.x_NMDA = self.integrator_x_NMDA(t, self.x_NMDA, [presyn_spikes, postsyn_spikes, 'rec_nmda'])
        self.s_NMDA = self.integrator_s_NMDA(t, self.s_NMDA,  [presyn_spikes, postsyn_spikes, 'rec_nmda_x' ,self.x_NMDA])
        return self.s_NMDA
    
    
    def I_NMDA(self, v, t):
        v_matrix = np.tile(np.expand_dims(v, 1), self.s_NMDA.shape[1])
        return self.parameter_matrices['rec_nmda']['g_rec_nmda']*(v_matrix-self.parameter_matrices['rec_nmda']['VE'])*(self.recurrent_connections[2]*self.s_NMDA) / (1 + self.parameter_matrices['rec_nmda']['Mg2+']*np.exp(-0.062*v/3.57))
    
    def d_s_NMDA(self, t, s_NMDA, extra_params):
        presyn_spikes, postsyn_spikes, mode, x = extra_params
        ds = -s_NMDA / self.parameter_matrices['rec_nmda']['tau_NMDA_decay'] + self.parameter_matrices['rec_nmda']['alpha_NMDA']*x*(np.ones_like(s_NMDA) - s_NMDA)
        return ds
    
    def d_x_NMDA(self, t, x_NMDA, extra_params):
        presyn_spikes, postsyn_spikes, mode = extra_params
        postsyn_trains = np.tile(np.expand_dims(postsyn_spikes, 0), (postsyn_spikes.shape[0],1))
        # is_spiked = np.tile(np.expand_dims(postsyn_spikes, 1), postsyn_spikes.shape[0])
        # dx= -x_NMDA / self.parameter_matrices['rec_nmda']['tau_NMDA_rise'] + np.where(is_spiked, postsyn_spikes, np.zeros_like(x_NMDA))
        dx= -x_NMDA / self.parameter_matrices['rec_nmda']['tau_NMDA_rise'] + postsyn_trains
        return dx
    
    ## GABA
    def I_GABA(self, v, t):
        # v_matrix = np.tile(np.expand_dims(v, 1), self.s_GABA.shape[1])
        v_matrix = np.tile(np.expand_dims(v, 0), (self.s_GABA.shape[0],1))
        return self.parameter_matrices['rec_gaba']['g_rec_gaba']*(v_matrix-self.parameter_matrices['rec_gaba']['VL'])*(self.recurrent_connections[3]*self.s_GABA)
    
    def d_s_GABA(self, t, s_GABA, extra_params):
        presyn_spikes, postsyn_spikes = extra_params
        # is_spiked = np.tile(np.expand_dims(postsyn_spikes, 1), postsyn_spikes.shape[0])
        postsyn_trains = np.tile(np.expand_dims(postsyn_spikes, 0), (postsyn_spikes.shape[0],1))
        # ds = -s_GABA / self.parameter_matrices['rec_gaba']['tau_GABA']  + np.where(is_spiked, postsyn_spikes, np.zeros_like(s_GABA))
        ds = -s_GABA / self.parameter_matrices['rec_gaba']['tau_GABA']  + postsyn_trains
        return ds
    
    def update_s_GABA(self, t, presyn_spikes, postsyn_spikes):
        self.s_GABA = self.integrator_GABA(t, self.s_GABA, [presyn_spikes, postsyn_spikes])
    
    def calculate_external_synaptic_currents(self, presyn_spikes, postsyn_spikes, v, time_step, t):
        self.update_s_ext_AMPA(t, presyn_spikes, postsyn_spikes)
        # self.s_ext_ampa_log.append(self.s_ext_AMPA)
        
        return np.sum(self.I_ext_AMPA(v, t), 1)
    
    def calculate_recurrent_synaptic_currents(self, presyn_spikes, postsyn_spikes, v, time_step, t):
        self.update_s_rec_AMPA(t, presyn_spikes, postsyn_spikes)
        self.update_s_NMDA(t, presyn_spikes, postsyn_spikes)
        self.update_s_GABA(t, presyn_spikes, postsyn_spikes)
        
        I_rec_ampa = np.sum(self.I_rec_AMPA(v, t), 1)
        I_rec_nmda = np.sum(self.I_NMDA(v, t), 1)
        I_rec_gaba = np.sum(self.I_GABA(v, t), 1)
        
        return I_rec_ampa, I_rec_nmda, I_rec_gaba
    
        
        

class COBASynapses:
    def __init__(self, params):
        self.dt = params['dt']
        self.VE = params['VE']
        self.no_input_neurons = params['no_input_neurons']
        
        if 'channel_stack' in params.keys():
            self.channel_stack = params['channel_stack']
        
        if 'sources' in params.keys():
            self.sources = params['sources']
        
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
            self.VL = params['VL']
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
        return self.g_GABA*(v-self.VL)*np.sum(self.w_GABA*self.s_GABA)
    
    def d_s_GABA(self, t, s_GABA, extra_params):
        presyn_spikes = extra_params[0]
        ds = -s_GABA / self.tau_GABA + np.where(self.spiked, presyn_spikes, np.zeros_like(s_GABA))
        return ds
    
    def update_s_GABA(self, t, presyn_spikes):
        #integrator(t,y0,extra_params as list)
        self.s_GABA = self.integrator_GABA(t, self.s_GABA, [presyn_spikes])
        return self.s_GABA
    