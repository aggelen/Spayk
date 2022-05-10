#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:57:30 2022

@author: aggelen
"""

class Tissue:
    def __init__(self):
        self.neurons = []
    
    def add(self, neurons):
        for n in neurons:
            self.neurons.append(n)
        
    def embody(self):
        pass
    