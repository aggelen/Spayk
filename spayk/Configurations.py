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
                'tau_ref': 2e-3,
                'visual_style': "\"k^\", markersize=10",
                "type": "pyramidal_cell"}

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
                'tau_ref': 1e-3,
                'visual_style': "\"r.\", markersize=20",
                "type": "interneuron"}

experimental_pyramidal_cell_params = {"g_ext_ampa": 2.1,
                                    "g_rec_ampa": 0.05,
                                    "g_rec_nmda": 0.165,
                                    "g_rec_gaba": 1.3}

experimental_interneuron_params = {"g_ext_ampa": 1.62,
                                   "g_rec_ampa": 0.04,
                                   "g_rec_nmda": 0.13,
                                   "g_rec_gaba": 1.0}




class SynapseConfigurator:
    def __init__(self, dt, neuron_type="standart"):
        self.dt = dt
        self.no_input_neurons = 0
        self.sources = {'ext_AMPA': [], 'rec_AMPA': [], 'rec_GABA': [], 'rec_NMDA': []}
        self.channel_stack = []
        self.configuration = {}
        self.neuron_type = neuron_type
        
        if self.neuron_type == "standart":
            self.g_ext_AMPA = 2.1
            self.g_rec_AMPA = 0.05
            self.g_NMDA = 0.165
            self.g_GABA = 1.0
        elif self.neuron_type == "pyramidal_cell":
            self.g_ext_AMPA = experimental_pyramidal_cell_params['g_ext_ampa']
            self.g_rec_AMPA = experimental_pyramidal_cell_params['g_rec_ampa']
            self.g_NMDA = experimental_pyramidal_cell_params['g_rec_nmda']
            self.g_GABA = experimental_pyramidal_cell_params['g_rec_gaba']
        elif self.neuron_type == "interneuron":
            self.g_ext_AMPA = experimental_interneuron_params['g_ext_ampa']
            self.g_rec_AMPA = experimental_interneuron_params['g_rec_ampa']
            self.g_NMDA = experimental_interneuron_params['g_rec_nmda']
            self.g_GABA = experimental_interneuron_params['g_rec_gaba']

        
    def create_external_AMPA_channel(self, source):
        self.no_input_neurons += 1
        
        self.channel_stack.append('ext_AMPA')
        self.sources['ext_AMPA'].append(source)
        
        self.configuration.update({'dt': self.dt,
                                   'no_input_neurons': self.no_input_neurons,
                                   'VE': 0,
                                   'g_ext_AMPA': self.g_ext_AMPA,
                                   'tau_AMPA': 2e-3,
                                   'sources': self.sources
                                   })
        
    def create_recurrent_AMPA_channel(self, source):
        self.no_input_neurons += 1
        
        self.channel_stack.append('rec_AMPA')
        self.sources['rec_AMPA'].append(source)
        
        self.configuration.update({'dt': self.dt,
                                   'no_input_neurons': self.no_input_neurons,
                                   'VE': 0,
                                   'g_rec_AMPA': self.g_rec_AMPA,
                                   'tau_AMPA': 2e-3,
                                   'sources': self.sources
                                   })
        
    def create_recurrent_NMDA_channel(self, source):
        self.no_input_neurons += 1
        
        self.channel_stack.append('rec_NMDA')
        self.sources['rec_NMDA'].append(source)
        
        self.configuration.update({'dt': self.dt,
                                   'no_input_neurons': self.no_input_neurons,
                                   'VE': 0,
                                   'g_NMDA': self.g_NMDA,
                                   'Mg2+': 1e-3,
                                   'alpha_NMDA': 0.5e-3,
                                   'tau_NMDA_rise': 2e-3,
                                   'tau_NMDA_decay': 100e-3,
                                   'sources': self.sources
                                   })
        
    def create_recurrent_GABA_channel(self, source):
        self.no_input_neurons += 1
        
        self.channel_stack.append('rec_GABA')
        self.sources['rec_GABA'].append(source)
        
        self.configuration.update({'dt': self.dt,
                                   'no_input_neurons': self.no_input_neurons,
                                   'VE': 0,
                                   'VL': -70e-3,
                                   'g_GABA': self.g_GABA,
                                   'tau_GABA': 5e-3,
                                   'sources': self.sources
                                   })
        
    def generate_config(self):
        self.configuration.update({'channel_stack': self.channel_stack})
        return self.configuration
