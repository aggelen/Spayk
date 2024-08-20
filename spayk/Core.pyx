#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:21:35 2024

@author: gelenag
"""

cimport numpy

def compute_np(array_1, array_2, a, b, c):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c