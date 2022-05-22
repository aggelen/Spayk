#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:49:06 2022

@author: aggelen
"""

class ConstantCurrentSource:
    def __init__(self, mA):
        self.mA = mA
    
    def I(self):
        return self.mA
    
class ExternalCurrentSignal:
    def __init__(self, signal):
        self.signal = signal
        self.idx = 0
        
    def I(self):
        I = self.signal[self.idx]
        self.idx += 1
        return I
        
        