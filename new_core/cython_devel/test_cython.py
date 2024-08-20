#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:39:35 2024

@author: gelenag
"""

import numpy as np
import generated_code

def test_cython_sum():
    numbers = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = generated_code.compute_sum(numbers)
    print(f"Sum using Cython: {result}")

if __name__ == "__main__":
    test_cython_sum()