#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:14:43 2024

@author: gelenag
"""
from numba import njit
import numpy as np

@njit(fastmath=True)
def poisson_generator_CONN_6():
    prob = np.random.uniform(0, 1, (1600, 1000))
    return np.less(prob, 2.4*0.01)

A = poisson_generator_CONN_6()