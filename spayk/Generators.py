#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 19:16:53 2022

@author: aggelen
"""
import numpy as np

class PoissonSpikeGenerator:
    def __init__(self, dt, seed=2022):
        self.dt = dt
        self.seed = seed
        
    def generate(self, t_end, no_neurons=1, firing_rate=10, use_seed=False):
        time_array = np.arange(0, t_end, self.dt)
        if use_seed:
            np.random.seed(seed=self.seed)
        
        u_rand = np.random.rand(no_neurons, time_array.shape[0])
        poisson_train = (u_rand < firing_rate * (self.dt / 1000.))
        
        return poisson_train.astype(int)