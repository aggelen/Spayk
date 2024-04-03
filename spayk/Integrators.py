#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:26:30 2024

@author: gelenag
"""

class Integrator:
    def __init__(self, dt, f):
        self.dt = dt
        self.f = f
    
    def __call__(self, t, y0, extra=None):
        return self.forward(t, y0, extra)
    
    def forward(self, params):
        pass
    
class EulerIntegrator(Integrator):
    def __init__(self, dt, f):
        super().__init__(dt, f)
        
    def forward(self, t, y0, extra=None):
        if extra is None:
            y1 = y0 + self.f(t, y0)*self.dt
        else:
            y1 = y0 + self.f(t, y0, extra)*self.dt
        return y1
    
        
class RK4Integrator(Integrator):
    def __init__(self, dt, f):
        super().__init__(dt, f)
    
    def forward(self, t, y0, extra=None):
        
        h = self.dt
        
        if extra is None:
            F1 = h*self.f(t,y0)
            F2 = h*self.f((t+h/2), (y0+F1/2))
            F3 = h*self.f((t+h/2),(y0+F2/2))
            F4 = h*self.f((t+h),(y0+F3))
        else:
            F1 = h*self.f(t,y0, extra)
            F2 = h*self.f((t+h/2), (y0+F1/2), extra)
            F3 = h*self.f((t+h/2),(y0+F2/2), extra)
            F4 = h*self.f((t+h),(y0+F3), extra)
            
        y1 = y0 + 1/6*(F1 + 2*F2 + 2*F3 + F4)
        return y1