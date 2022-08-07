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
from spayk.Stimuli import ConstantCurrentSource
from spayk.Utils import izhikevich_dynamics_selector

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# create izhikevich neurons with different dynamics
# if no parameters are given, all neurons are regular spiking by default.
# dynamics parameters can given as A,B,C,D matrices.
# there is a built-in dynamics selector!
n_neurons = 100
#randomly selected dynamics 0-6 (regular, int_burst, chatterng, fast, thalamo, reson, low_threshold)
A,B,C,D = izhikevich_dynamics_selector(np.random.randint(0,7,n_neurons))
group_params = {'no_neurons': n_neurons,
                'A': A,
                'B': B,
                'C': C,
                'D': D}
neurons = IzhikevichNeuronGroup(group_params)

# bind neuron groups to a tissue
test_tissue = Tissue([neurons])

# run simulation
stim_params = {'t_stop': 1000,
               'dt': 0.1,
               't_cur_start': np.random.randint(100, 400, group_params['no_neurons']),
               't_cur_end': np.random.randint(600, 900, group_params['no_neurons']),
               'amplitudes': np.full(group_params['no_neurons'], 5)}

test_tissue.keep_alive(stimuli=ConstantCurrentSource(stim_params))
test_tissue.logger.plot_v(np.random.randint(0,n_neurons,5))
test_tissue.logger.raster_plot()