#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:47:17 2022

@author: aggelen
"""

import sys
sys.path.append('..')

from spayk.Organization import Tissue
from spayk.Models import SRMLIFNeuronGroup
from spayk.Stimuli import SpikingMNIST

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


#%% Stimuli
dataset = SpikingMNIST()

# for every label (0-9), find weights via stdp

