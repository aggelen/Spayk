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
import pickle
from tqdm import tqdm
from problem import Problem

class SpaykCore:
    def __init__(self):
        self.ready = False
        self.codegen = CodeGenerator()
        self.code_as_string =""""""       
        
    def keep_alive(self, tsim):
        self.ready = True
        self.time = np.arange(0.0, tsim, self.params['dt'])
        if self.ready:
            print("Sim. [bold magenta]Started[/]!") 
            
            # loc = {}
            # exec(self.code_as_string, globals(), loc)
            # self.problem = loc['problem']
            
            self.problem = Problem()
            
            t_idx = 0
            for t in tqdm(self.time):
                self.problem.forward(t_idx)
                t_idx += 1
        else:
            print("Error: Sim. [bold magenta]Stopped[/]!") 
        
        # solver solve

class NeuralCircuit(SpaykCore):
    def __init__(self, neurons, synapses, stimulus, params):
       super().__init__()      
       self.params = params
       
       self.neurons = neurons
       self.synapses = synapses
       self.stimulus = stimulus
       
       self.codegen.analyze_network(neurons, synapses, stimulus, params)
       self.code_as_string = self.codegen.make_sysofodes(neurons, synapses, stimulus)
    
        
class NeuralNetwork:
    def __init__(self):
        pass
    
