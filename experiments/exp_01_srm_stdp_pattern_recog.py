#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:47:17 2022

@author: aggelen
"""

import sys
sys.path.append('..')

from spayk.Organization import Tissue
from spayk.Models import IzhikevichNeuronGroup


recognation_tissue = Tissue([])

#%% Simulations
# run simulation
sim_params = {'dt': 0.1}
recognation_tissue.keep_alive(sim_params)