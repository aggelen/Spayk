#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:03:16 2024

@author: gelenag
"""

class Wang2002Config:
    dt = 0.1e-3
    no_neurons = 2000
    no_exc = 1600
    no_A = 240
    no_B = 240
    no_N = 1120
    no_inh = 400
    
    #%% Noise and stim params
    no_noise_E = 1000
    no_noise_I = 1000
    no_stim_A = 240
    no_stim_B = 240
    
    freq_noise_E = 2.4
    freq_noise_I = 2.4
    freq_stim_A = 15
    freq_stim_B = 30
    
    stim_time_params = (0, 1, dt)
    
    #%% 
    assert(no_A+no_B+no_N == no_exc)
    assert(no_exc+no_inh == no_neurons)
    
    f = 0.15 
    w_plus = 1.7  # relative synaptic strength inside a selective population (1.0: no potentiation))
    w_minus = 1.0 - f * (w_plus - 1.0) / (1.0 - f)
    
    #%% Neuron parameters
    VE = 0
    VI = -70e-3
    VL = -70.0e-3  # resting potential
    VT = -50.0e-3  # firing threshold
    VR = -55.0e-3  # reset potential
    CM_E = 0.5e-9  # membrane capacitance for pyramidal cells (excitatory neurons)
    CM_I = 0.2e-9  # membrane capacitance for interneurons (inhibitory neurons)
    GL_E = 25.0e-9  # membrane leak conductance of excitatory neurons
    GL_I = 20.0e-9  # membrane leak conductance of inhibitory neurons
    REF_E = 2.0e-3  # refractory periodof excitatory neurons
    REF_I = 1.0e-3  # refractory period of inhibitory neurons
    
    # exc_neuron_params = {'VE': VE, 'VI': VI, 'VL': VL, 'VT': VT, 'VR': VR, 'CM': CM_E, 'GL': GL_E, 'TREF': REF_E}
    neuron_params = {'VE': VE, 'VI': VI, 'VL': VL, 'VT': VT, 'VR': VR, 'CM': CM_E, 'GL': GL_E, 'TREF': REF_E}
    
    g_ampa_ext2exc = 2.1*1e-9  # external -> excitatory (AMPA)
    g_ampa_ext2inh = 1.62*1e-9  # external -> inhibitory neurons (AMPA)
    g_ampa_exc2exc = 0.05*1e-9 / no_exc * 1600  # excitatory -> excitatory neurons (AMPA)
    g_ampa_exc2inh = 0.04*1e-9 / no_exc * 1600  # excitatory -> inhibitory neurons (AMPA)
    g_nmda_exc2exc = 0.165*1e-9 / no_exc * 1600  # excitatory -> excitatory neurons (NMDA)
    g_nmda_exc2inh = 0.13*1e-9 / no_exc * 1600  # excitatory -> inhibitory neurons (NMDA)
    g_gaba_inh2exc = 1.3*1e-9 / no_inh * 400  # inhibitory -> excitatory neurons (GABA)
    g_gaba_inh2inh = 1.0*1e-9 / no_inh * 400  # inhibitory -> inhibitory neurons (GABA)
 
    synapse_params = {'tau_AMPA': 2.0e-3,
                      'tau_NMDA_rise': 2.0e-3,
                      'tau_NMDA_decay': 100.0e-3, 
                      'tau_GABA': 5.0e-3,
                      'alpha': 0.5e3,
                      'C_Mg': 1e-3}   
 
    # exc_neuron_params = {'VL': VL,
    #                      'VT': VT, 
    #                      'VR': VR, 
    #                      'CM': CM_E,
    #                      'GL': GL_E, 
    #                      'REF': REF_E,
    #                      'g_ampa_ext':2.1*1e-9,
    #                      'g_ampa_rec': 0.05*1e-9 / no_exc * 1600, 
    #                      'g_nmda_rec': 0.165*1e-9 / no_exc * 1600, 
    #                      'g_gaba_rec': 1.3*1e-9 / no_inh * 400,
    #                      'tau_AMPA': 2.0e-3,
    #                      'tau_NMDA_rise': 2.0e-3,
    #                      'tau_NMDA_decay': 100.0e-3, 
    #                      'tau_GABA': 5.0e-3,
    #                      'alpha': 0.5e3,
    #                      'C_Mg': 1e-3}
    
    # inh_neuron_params = {'VL': VL, 
    #                      'VT': VT, 
    #                      'VR': VR, 
    #                      'CM': CM_I, 
    #                      'GL': GL_I, 
    #                      'REF': REF_I,
    #                      'g_ampa_ext': 1.62*1e-9,
    #                      'g_ampa_rec': 0.04*1e-9 / no_exc * 1600,
    #                      'g_nmda_rec': 0.13*1e-9 / no_exc * 1600,
    #                      'g_gaba_rec': 1.0*1e-9 / no_inh * 400,
    #                      'tau_AMPA': 2.0e-3,
    #                      'tau_NMDA_rise': 2.0e-3,
    #                      'tau_NMDA_decay': 100.0e-3, 
    #                      'tau_GABA': 5.0e-3,
    #                      'alpha': 0.5e3,
    #                      'C_Mg': 1e-3}
    