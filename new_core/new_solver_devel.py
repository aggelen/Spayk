#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:03:20 2024

@author: gelenag
"""

from spayk.Solvers import euler
import numpy as np
import matplotlib.pyplot as plt


def func(x, t, dxdt):
    dxdt[0] = x[1]
    dxdt[1] = - x[0]
    
x0 = np.array([0., 1.])
t = np.arange(0, 10, 0.01)

X = euler(func, x0, t)

plt.plot(t, X)