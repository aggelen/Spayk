#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:44:35 2022

@author: aggelen
"""
import numpy as np

import sys
sys.path.append('..')

from spayk.Learning import STDP

stdp_params = {'a_plus': 0.03125, 'a_minus': 0.028, 'tau_plus': 16.8, 'tau_minus': 33.7}
time_difference = np.linspace(-100,100,100)

plasticity = STDP(stdp_params)
plasticity.plot_stdp_curve(time_difference)
