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
        