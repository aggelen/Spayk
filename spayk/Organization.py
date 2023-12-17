#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organization
====================================
The core module of my example project

"""

"""
Created on Tue May 10 21:57:30 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import seaborn as sns
import matplotlib
import itertools

class Logger:
    def __init__(self, steps, dt):
        """

        Parameters
        ----------
        steps : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.steps = steps
        self.dt = dt
        self.v_history = []
        self.I_history = []
        
    def log(self, v, I):
        self.v_history.append(v)
        self.I_history.append(I)
        
    def log_v(self, v):
        self.v_history.append(v)
        
    def plot_v(self, sel_idx=None):  
        if isinstance(self.v_history, list):
            vs = np.array(self.v_history, dtype=np.float32)
        else:
            vs = self.v_history
            
        if vs.shape.__len__() == 1:
            vs = np.expand_dims(vs, axis=1)
            
        if sel_idx is not None:
            vs_sel = vs.T[sel_idx]
            fig, axes = plt.subplots(nrows=sel_idx.__len__(), ncols=1, sharex=True)
            i = 0
            for ax in axes:
                ax.plot(np.arange(vs_sel.shape[1])*self.dt, vs_sel[i])
                ax.set_title("Neuron #" + str(i))
                i += 1
            # plt.tight_layout()
        else:
            fig, axes = plt.subplots(nrows=vs.shape[1], ncols=1, sharex=True)
            if isinstance(axes, np.ndarray):
                i = 0
                for ax in axes:
                    ax.plot(np.arange(vs.shape[0])*self.dt, vs.T[i])
                    ax.set_title("Neuron #" + str(i))
                    i += 1
            else:
                axes.plot(np.arange(vs.shape[0])*self.dt, vs)
            # plt.tight_layout()
            
        plt.xlabel('Time (ms)')
        plt.ylabel('Memb. Pot. (mV)')
        # fig.text(0.5, 0.04, 'Time (ms)', ha='center')
        # fig.text(0.04, 0.5, 'Memb. Pot. (unitless)', va='center', rotation='vertical')
            
    def raster_plot(self, color_array=None, title=None, title_pad=0):
        f = plt.figure()
        if title is not None:
            plt.title(title, pad=title_pad)
        else:
            plt.title('Raster Plot', pad=title_pad)
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        # spike_times = []
        spikes = (np.array(self.v_history) == 35.0).T
        
        mean_spike_rate = np.sum(spikes,1).mean()
        print('Output Mean Spike Rate: {}'.format(mean_spike_rate))
        
        spike_loc = np.argwhere(spikes)
        
        sns.set()
        palette = itertools.cycle(sns.color_palette())
        
        if color_array is not None:
            # c = plt.cm.Set1(color_array[spike_loc[:,0]])
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            c = []
            for k in color_array[spike_loc[:,0]]:
                c.append(colors[int(k)])

        else:
            c = 'k'
            
        plt.scatter(spike_loc[:,1]*self.dt, spike_loc[:,0], s=3, color=c)
        
        # for spike_train in spikes:
        #     spike_times.append(np.where(spike_train)[0]*self.dt)
            
        # if color_array is not None:
        #     # c = plt.cm.Set1(color_array)
        #     c = []
        #     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        #     for i in color_array:
        #         c.append(colors[int(i)])
        # else:
        #     c = 'k'
            
        # plt.eventplot(spike_times, colors=c, linewidths=2.5)
        # plt.xlim([0, int(spikes.shape[1]*self.dt)])
        
        # ax = f.gca()
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def output_spikes(self):
        spikes = (np.array(self.v_history) == 35.0).T
        return spikes
        

class Tissue:
    def __init__(self, neuron_groups=None, params=None):
        # self.dt = params['dt']
        #FIXME! single group
        self.neuron_group = neuron_groups[0]
        
        self.output = None
        
    def keep_alive(self, stimuli):
        self.neuron_group.prepare(dt=stimuli.dt)
        
        if stimuli.source_type == 'current':
            self.logger = Logger(stimuli.steps, stimuli.dt)
            for step in tqdm(stimuli.steps):
                t = step*stimuli.dt
                inj_current = stimuli.I()
                self.neuron_group.inject_current(inj_current)
                v, _ = self.neuron_group()
                self.logger.log(v, inj_current)
        elif stimuli.source_type == 'spike_train':
            self.logger = Logger(stimuli.steps, stimuli.dt)
            for step in tqdm(stimuli.steps):
                t = step*stimuli.dt
                input_spikes = stimuli.current_spikes()
                                
                out = self.neuron_group(input_spikes)
                if isinstance(out, np.float64):
                    self.logger.log_v(out)
                else:
                    self.logger.log_v(out[0])
        elif stimuli.source_type == 'spike_instance':
             input_spikes = stimuli.current_spikes
             self.output = self.neuron_group(input_spikes)
        else:
            raise Exception('Invalid Stimuli Source Type')

# class Tissue:
#     def __init__(self, connected=None, stdp_status=True, name=''):
#         self.neurons = []
#         self.name = name
#         self.stdp_status = stdp_status
#         self.vs = []
#         self.us = []
#         self.Is = []
        
#         self.dynamics_matrix = []
#         self.neuron_modes = []
#         self.stim_type = []
        
#         self.log_v = []
#         self.log_u = []
#         self.log_spikes = []
#         self.log_Is = []
        
#         self.LTD_history = np.zeros((0,1))
#         self.gE_bar_update = np.ones((0,1))
        
        
#         if connected is not None:
#             self.is_connected = True
#             self.connected = connected
#         else:
#             self.is_connected = False
        
        

#     def add(self, neurons):
#         for n in neurons:
#             self.neurons.append(n)
            
#             #FIXME! Initialization
#             self.vs.append(-65)
#             self.us.append(-13)
#             self.Is.append(n.stimuli.I)
            
#             stim_type = type(n.stimuli).__name__
#             if stim_type == 'ConstantCurrentSource' or stim_type == 'ExternalCurrentSignal':
#                 self.stim_type.append(0)
#             elif stim_type == 'GENESIS_Synapse':
#                 self.stim_type.append(1)
                
#             self.dynamics_matrix.append([n.a,n.b,n.c,n.d,n.vt])
#             self.neuron_modes.append(n.mode)
            
#             self.LTD_history = np.r_[self.LTD_history, np.array([[0]])]
#             # self.gE_bar_update = np.r_[self.gE_bar_update, np.array([[0.024]])]
            
#     def calculate_Is(self, t, synaptic_plasticity=False):
#         Is = []
#         for i, I in enumerate(self.Is):
#             if self.stim_type[i] == 0:  #const source
#                 Is.append(I())
#             elif self.stim_type[i] == 1:    #genesis synapse
#                 if len(self.log_spikes) == 0:
#                     spikes = np.zeros_like(self.vs)
                    
#                     #FIXME!
#                     if self.is_connected:
#                         connection_spikes = np.zeros_like(self.connected[0].vs)
#                     else:
#                         connection_spikes = None
#                 else:
#                     spikes = self.log_spikes[-1].astype(int)
#                     if self.is_connected:
#                         connection_spikes = self.connected[0].log_spikes[-1].astype(int)  
#                     else:
#                         connection_spikes = None
                    
#                 Is.append(I(i, self.vs, spikes, t, connection_spikes, synaptic_plasticity))
#         return Is
            
#     def LTD_update(self, spikes, dt, tau_stdp=20, A_minus=0.008*1.10):
#         if any(spikes):
#             self.LTD_history[spikes,-1] = self.LTD_history[spikes,-1] - A_minus
            
#             spiking_neurons = [i for (i, v) in zip(self.neurons, spikes) if v] 
#             for neuron in spiking_neurons:
#                 neuron.stimuli.update_gebar_with_postspike()
            
#         self.LTD_history = np.c_[self.LTD_history, 
#                                  np.copy(self.LTD_history[:,-1] - dt / tau_stdp * self.LTD_history[:,-1])]
        
            
#     def embody(self):
#         self.vs = np.array(self.vs)
#         self.us = np.array(self.us)
#         self.Is = np.array(self.Is)
#         self.dynamics_matrix = np.array(self.dynamics_matrix)
    
#     def keep_log(self, spikes, currents):
#         self.log_v.append(self.vs)
#         self.log_u.append(self.us)
#         self.log_spikes.append(spikes)
#         self.log_Is.append(currents)
        
#     def end_of_life(self):
#         self.log_v = np.array(self.log_v)
#         self.log_u = np.array(self.log_u)
#         self.log_spikes = np.array(self.log_spikes, dtype=bool).T
#         self.log_Is = np.array(self.log_Is).T
        
#     def plot_membrane_potential_of(self, neuron_id, dt=0.1, hold_on=False, color=None):
#         time = np.arange(self.log_v.shape[0])*dt
#         if not hold_on:
#             plt.figure()
#         if color is not None:
#             plt.plot(time, self.log_v[:, neuron_id], color)
#         else:
#             plt.plot(time, self.log_v[:, neuron_id])
#         # plt.xlabel('Time (ms)')
#         plt.ylabel('mV')
#         plt.title('Membrane Potential of Neuron#{}'.format(neuron_id))
#         plt.grid()
#         plt.xlim([0, int(self.log_v.shape[0]*dt)])
        
#     def plot_current_of_neuron(self, neuron_id, dt, hold_on=False):
#         time = np.arange(self.log_v.shape[0])*dt
#         if not hold_on:
#             plt.figure()
#         plt.plot(time, self.log_Is[neuron_id])
#         plt.xlabel('Time (ms)')
#         plt.ylabel('mA')
#         plt.title('Input Current of Neuron#{}'.format(neuron_id))
#         plt.grid()
        
#     def raster_plot(self, dt=0.1):
#         # color_list = [ "red", "blue", "green", "yellow", "purple", "orange", "white", "black" ]
#         color_list = plt.get_cmap('tab10')
#         colors = [color_list(mode) for mode in self.neuron_modes]
        
#         labels = ['regular_spiking',
#                   'intrinsically_bursting', 
#                   'chattering', 
#                   'fast_spiking', 
#                   'thalamo_cortical',
#                   'resonator',
#                   'low_threshold_spiking']
        
#         f = plt.figure()
#         plt.title('Raster Plot')
#         plt.xlabel('Time (ms)')
#         plt.ylabel('Neuron ID')
#         spike_times = []
#         for spike_train in self.log_spikes:
#             spike_times.append(np.where(spike_train)[0]*dt)
#         ep = plt.eventplot(spike_times, colors=colors)
#         plt.xlim([0, int(self.log_spikes.shape[1]*dt)])
        
#         ax = f.gca()
#         ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#         # plt.legend(labels, colors)