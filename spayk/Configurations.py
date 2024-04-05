#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:05:55 2024

@author: gelenag
"""

# %% Configurations for specific cell types


class NeuronConfigurator:
    def __init__(self):
        pass

    @staticmethod
    def lif_pyramidal_cell(dt):
        # %% Pyramidal Neuron Parameters (Wang2002)
        # VL: resting potential (mV), Vth: Firing Threshold (mV) Vreset: Reset Potential (mV), Cm: Membrane Cap. (F), gL: Memb. Leak. Condctance (S)
        # tau_ref: Refractory Period (ms),
        return {'dt': dt,
                'VL': -70e-3,
                'Vth': -50e-3,
                'Vreset': -55e-3,
                'Cm': 0.5e-9,
                'gL': 25e-9,
                'tau_ref': 2e-3}

    @staticmethod
    def lif_interneuron(dt):
        # %% An interneuron (Wang2002)
        # VL: resting potential (mV), Vth: Firing Threshold (mV) Vreset: Reset Potential (mV), Cm: Membrane Cap. (F), gL: Memb. Leak. Condctance (S)
        # tau_ref: Refractory Period (ms),
        return {'dt': dt,
                'VL': -70e-3,
                'Vth': -50e-3,
                'Vreset': -55e-3,
                'Cm': 0.2e-9,
                'gL': 20e-9,
                'tau_ref': 1e-3}


class SynapseConfigurator:
    def __init__(self, dt):
        self.dt = dt
        self.no_input_neurons = 0
        self.sources = {'ext_AMPA': []}
        self.channel_stack = []
        self.configuration = {}
        
    def create_external_AMPA_channel(self, source):
        self.no_input_neurons += 1
        
        self.channel_stack.append('ext_AMPA')
        self.sources['ext_AMPA'].append(source)
        
        self.configuration.update({'dt': self.dt,
                                   'no_input_neurons': self.no_input_neurons,
                                   'VE': 0,
                                   'g_ext_AMPA': 2.1,
                                   'tau_AMPA': 2e-3,
                                   'sources': self.sources
                                   })
        
    def generate_config(self):
        self.configuration.update({'channel_stack': self.channel_stack})
        return self.configuration
