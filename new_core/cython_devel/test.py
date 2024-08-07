#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:07:49 2024

@author: gelenag
"""

from math import sin


def f(x):
    return sin(x**2)


def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx


N = 8_000_000

integrate_f(0, 2, N)