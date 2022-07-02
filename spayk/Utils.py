#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:27:52 2022

@author: aggelen
"""
import numpy as np

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

