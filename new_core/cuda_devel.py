#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:43:50 2024

@author: gelenag
"""

from numba import cuda

@cuda.jit
def add_gpu(x, out):
    idx = cuda.grid(1)
    out[idx] = x[idx] + 2
    
import numpy as np

a = np.arange(10,dtype=np.float32)

# send input vector to the device
d_a = cuda.to_device(a)

# create output vector on the device
d_out = cuda.device_array_like(d_a)# we decide to use 2 blocks, each containing 5 threads for our vector 
nbr_block_per_grid = 2
nbr_thread_per_block = 5
add_gpu[nbr_block_per_grid, nbr_thread_per_block](d_a, d_out)# now we get our output
out = d_out.copy_to_host()
out