#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:34:27 2024

@author: gelenag
"""

import matplotlib.pyplot as plt
import numpy as np





class ODESolver:
    def __init__(self):
        pass


# %% Integrators


class EulerIntegrator:
    def __init__(self):
        pass


class LIFIntegrator(EulerIntegrator):
    def __init__(self):
        super().__init__()

    def lif(u, p, t):
        gL, EL, C, Vth, I = p
        return (-gL*(u-EL)+I)/C

    def threshold(u, t, integrator):
        integrator.u > integrator.p[4]

    def reset(integrator):
        integrator.u = integrator.p[2]


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
