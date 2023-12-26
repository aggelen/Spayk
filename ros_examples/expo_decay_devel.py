#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:23:10 2023

@author: gelenag
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 1/30
spikes = np.random.randint(0, 2, 1)
presyn_traces = np.zeros_like(spikes)
to_pre = 3*(1/30)

spik = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]

trace_log = []
for s in spik:
    spikes = s
    
    d_apre = -presyn_traces / to_pre
    apre = presyn_traces + d_apre*dt
    
    presyn_traces = np.where(spikes == 1, np.ones_like(presyn_traces), apre)
    
    trace_log.append(presyn_traces)
    
    
trace_log = np.array(trace_log)
plt.plot(trace_log)