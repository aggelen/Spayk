#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 00:24:08 2024

@author: gelenag
"""

import numpy as np
from spayk.Stimuli import *

tsim = 4
v = -55e-3
t_ref = 0
dt = 0.1e-3
d_V = 0
output_spikes = []
output_v = []
t = np.arange(0,tsim,dt)
vl = -70e-3

gampa = 2.1e-9

I_syn_hist = []
noise = PoissonSpikeTrain(1000, 2.4, (0, tsim, dt))
s_AMPA_EXT = np.zeros((1,1000))
d_s_AMPA_EXT = np.zeros((1,1000))

s_AMPA_EXT_hist = []

for tid, ts in enumerate(t):
    d_s_AMPA_EXT = (-s_AMPA_EXT / 0.002) 

    s_AMPA_EXT = s_AMPA_EXT + d_s_AMPA_EXT*dt
    
    s_AMPA_EXT = s_AMPA_EXT + gampa*noise.spikes[:,tid]

    I_syn = (v)*np.sum(s_AMPA_EXT, axis=1)

    s_AMPA_EXT_hist.append(np.sum(s_AMPA_EXT, axis=1))

    I_syn_hist.append(I_syn)

    is_in_rest = np.greater(t_ref, 0.0)
    t_ref = np.where(is_in_rest, t_ref - dt, t_ref)

    d_V = np.where(is_in_rest, 0, (-25.0e-9*(v-vl) - I_syn) / 0.5e-9)
    integrated_V = v + d_V*dt

    v = np.where(np.logical_not(is_in_rest), integrated_V, v)
    is_fired = np.greater_equal(v, -50e-3)
    v = np.where(is_fired, -55e-3, v)
    t_ref = np.where(is_fired, 2e-3, t_ref)

    output_spikes.append(np.copy(is_fired))
    output_v.append(np.copy(v))

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(output_spikes)

plt.figure()
plt.plot(I_syn_hist)
plt.title('I_syn_hist')

s_AMPA_EXT_hist = np.array(s_AMPA_EXT_hist)
plt.figure()
plt.plot(s_AMPA_EXT_hist)
plt.title('s_AMPA_EXT_hist')

print("Firing Rate: {}".format(np.sum(output_spikes)/4))