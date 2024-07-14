#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:34:27 2024

@author: gelenag
"""

class ProblemGenerator:
    def __init__(self):
        pass

class Solver:
    def __init__(self):
        pass
    
    
#%% Integrators

import numpy as np
import matplotlib.pyplot as plt

def euler(f, x0, t):
    X = np.zeros((len(t), len(x0)), float)
    dxdt = np.zeros(len(x0), float)

    X[0, :] = x = np.array(x0)
    tlast = t[0]

    for n, tcur in enumerate(t[1:], 1):
        f(x, tlast, dxdt)
        X[n, :] = x = x + dxdt * (tcur - tlast)
        tlast = tcur

    return X


