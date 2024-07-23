#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:03:20 2024

@author: gelenag
"""

from spayk.Solvers import euler
import numpy as np
import matplotlib.pyplot as plt

import time

# # def func(x, t, dxdt):
# #     dxdt[0] = x[1]
# #     dxdt[1] = - x[0]
    
x0 = np.array([0., 1.])
t = np.arange(0, 10, 0.01)

# # X = euler(func, x0, t)


from spayk.CIntegrators import euler




#%%
def pyeuler(f, x0, t):
    X = np.zeros((len(t), len(x0)), float)
    dxdt = np.zeros(len(x0), float)

    X[0, :] = x = np.array(x0)
    tlast = t[0]

    for n, tcur in enumerate(t[1:], 1):
        f(x, tlast, dxdt)
        X[n, :] = x = x + dxdt * (tcur - tlast)
        tlast = tcur

    return X

def func(x, t, dxdt):
    dxdt[0] = x[1]
    dxdt[1] = - x[0]

#%%
time_start = time.process_time()

X = pyeuler(func, x0, t)

time_end = time.process_time()

time_duration = time_end - time_start
# report the duration
print(f'Took {time_duration:.6f} seconds')
# plt.plot(t, X)
    
#%%
time_start = time.process_time()
X2 = euler(x0, t)
time_end = time.process_time()

time_duration = time_end - time_start
# report the duration
print(f'Took {time_duration:.6f} seconds')


plt.plot(t, X)