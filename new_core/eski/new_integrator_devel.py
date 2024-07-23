#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:51:45 2024

@author: gelenag
"""

def lif(u,p,t):
    gL, EL, C, Vth, I = p
    return (-gL*(u-EL)+I)/C

def threshold(u, t, integrator):
    integrator.u > integrator.p[4]

def reset(integrator):
    integrator.u = integrator.p[2]
    
def 
    
u0 = -75
tspan = (0.0, 40.0)
# p = (gL, EL, C, Vth, I)
p = [10.0, -75.0, 5.0, -55.0, 0]

