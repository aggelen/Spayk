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

import matplotlib.pyplot as plt
plt.close('all')

# create izhikevich neurons with different dynamics
# if no parameters are given, all neurons are regular spiking by default.
# dynamics parameters can given as A,B,C,D matrices.
# there is a built-in dynamics selector!

# A,B,C,D = izhikevich_dynamics_selector()
group_params = {'no_neurons': 5}
neurons = IzhikevichNeuronGroup(group_params)

# bind neuron groups to a tissue
sim_params = {}

test_tissue = Tissue([neurons], sim_params)

# run simulation
import numpy as np
stim_params = {'t_stop': 1000,
               'dt': 0.1,
               't_cur_start': np.random.randint(100, 400, group_params['no_neurons']),
               't_cur_end': np.random.randint(600, 900, group_params['no_neurons']),
               'amplitudes': np.random.randint(3, 10, group_params['no_neurons'])}

injected_currents = ConstantCurrentSource(stim_params)
injected_currents.plot()

test_tissue.keep_alive(stimuli=injected_currents)
test_tissue.logger.plot_v()