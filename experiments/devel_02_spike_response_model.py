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
from spayk.Stimuli import PoissonSpikeTrain

# create two neurons as a group with LIF/SRM model
group_params = {'no_neurons': 2}
srm_neuron_group = SRMLIFNeuronGroup(group_params)

# bind neuron groups to a tissue
test_tissue = Tissue([srm_neuron_group])

input_spike_train = PoissonSpikeTrain(dt=0.1, t_stop=1000, no_neurons=2, spike_rates=[20,35])
input_spike_train.raster_plot()

# run simulation
test_tissue.keep_alive(stimuli=input_spike_train)
# test_tissue.logger.plot_v(np.random.randint(0,n_neurons,5))
# test_tissue.logger.raster_plot()