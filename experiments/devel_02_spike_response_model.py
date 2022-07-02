#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:47:17 2022

@author: aggelen
"""

import sys
sys.path.append('..')

from spayk.Organization import Tissue
from spayk.Models import SRMLIFNeuron
from spayk.Stimuli import PoissonSpikeTrain

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# create two neurons as a group with LIF/SRM model
group_params = {'n_synapses': 10,
                'dt': 1.0}

srm_neuron_group = SRMLIFNeuron(group_params)

# bind neuron groups to a tissue
test_tissue = Tissue([srm_neuron_group])

input_spike_train = PoissonSpikeTrain(dt=1.0, t_stop=1000, no_neurons=10, spike_rates=np.random.randint(10,60,10))
input_spike_train.raster_plot()

# run simulation
test_tissue.keep_alive(stimuli=input_spike_train)
test_tissue.logger.plot_v()
# test_tissue.logger.raster_plot()