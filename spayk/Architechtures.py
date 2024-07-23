#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:15:05 2024

@author: gelenag
"""
import numpy as np
from rich import print

from spayk.Solvers import ProblemGenerator, ODESolver

class SpaykCore:
    def __init__(self):
        self.ready = False
        
        self.problem_generator = ProblemGenerator()
        
        self.problem = None
        self.solver = None
        
    def set_problem(self, problem):
        self.problem = problem

    def set_solver(self, solver):
        self.solver = solver
        
        
    def keep_alive(self, tsim):
        self.ready = True
        if self.ready:
            print("Sim. [bold magenta]Started[/]!") 
        else:
            print("Error: Sim. [bold magenta]Stopped[/]!") 
        
        # solver solve

class NeuralCircuit(SpaykCore):
    def __init__(self, neurons, synapses, stimulus):
       super().__init__()      
       
       
       self.neurons = neurons
       self.synapses = synapses
       self.stimulus = stimulus
       
       self.problem_generator.analyze_network(neurons, synapses, stimulus)
       
       problem = self.problem_generator.make_sysofodes(neurons, synapses, stimulus)
       
       self.set_problem(problem)
       self.set_solver(ODESolver())
    
    
        
class NeuralNetwork:
    def __init__(self):
        pass
    
