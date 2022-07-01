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
from spayk.Stimuli import ExternalSpikeTrain

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# create two neurons as a group with LIF/SRM model
group_params = {'no_neurons': 1,
                'n_synapses': 1,
                'dt': 1.0,
                'v_th': 1.0}

srm_neuron_group = SRMLIFNeuronGroup(group_params)

# bind neuron groups to a tissue
test_tissue = Tissue([srm_neuron_group])
st = np.zeros(80)
st[[2, 23, 44, 45, 48, 61]] = 1
input_spike_train = ExternalSpikeTrain(dt=1.0, t_stop=80, no_neurons=1, spike_train=st)

# run simulation
test_tissue.keep_alive(stimuli=input_spike_train)
test_tissue.logger.plot_v()
# test_tissue.logger.raster_plot()