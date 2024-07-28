#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:15:05 2024

@author: gelenag
"""
import numpy as np
from rich import print

from spayk.Core import CodeGenerator
from spayk.Stimuli import *

class SpaykCore:
    def __init__(self):
        self.ready = False
        self.codegen = CodeGenerator()
        self.code_as_string =""""""       
        
    def keep_alive(self, tsim):
        self.ready = True
        if self.ready:
            print("Sim. [bold magenta]Started[/]!") 
            
            # loc = {}
            # exec(self.code_as_string, globals(), loc)
            # self.problem = loc['problem']
        else:
            print("Error: Sim. [bold magenta]Stopped[/]!") 
        
        # solver solve

class NeuralCircuit(SpaykCore):
    def __init__(self, neurons, synapses, stimulus):
       super().__init__()      
       
       
       self.neurons = neurons
       self.synapses = synapses
       self.stimulus = stimulus
       
       self.codegen.analyze_network(neurons, synapses, stimulus)
       
       self.code_as_string = self.codegen.make_sysofodes(neurons, synapses, stimulus)
    
        
class NeuralNetwork:
    def __init__(self):
        pass
    
