#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:08:47 2022

@author: aggelen
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class STDP:
    def __init__(self, params):
        self.params = params
        self.effective_time_window = 70
        self.mu_w_hist = []
        
    def single_LTP_update(self, dt):
        return self.params['a_plus'] * np.exp((dt/self.params['tau_plus']))
        
    def single_LTD_update(self, dt):
        return -self.params['a_minus'] * np.exp(-(dt/self.params['tau_minus']))
    
    def plot_stdp_curve(self, time_difference):
        dw = []
        for dt in time_difference:
            if dt <= 0:
                dw.append(self.single_LTP_update(dt))
            else:
                dw.append(self.single_LTD_update(dt))
        
        plt.figure()
        plt.plot(time_difference, dw)
        plt.xlabel(u'Δt Time Difference (ms)')
        plt.ylabel(u'Δw Relative Weight Change')
        plt.grid()
        
               
        
        
    def update_spike_times(self, t_ti, t_tj, current_spikes):
        t_ti += self.params['dt']
        t_tj += self.params['dt']

        new_spike_times = np.where(current_spikes,
                                   np.full(current_spikes.size, 0.0),
                                   np.full(current_spikes.size, 100000.0))
        
        t_tj = np.r_[[new_spike_times], t_tj]
        t_tj = np.delete(t_tj, -1, 0)

        return t_ti, t_tj
        
    def offline(self, pre_spikes, post_spikes, weights, log_changes=False, learning_rate=5e-3):
        # presyn_spike_time -> tj, postsyn_spike_time -> ti
        # for neuron_spiked in post_spikes:
        #!! single neuron
        t_ti = 1e3
        t_tj = np.full((self.effective_time_window, pre_spikes.shape[0]), 100000.0)
        presyn_spikes = np.full(pre_spikes.shape[0], False)
        
        self.learning_rate = learning_rate
        
        if log_changes:
            self.log_changes = True
            self.log_delta_w = np.zeros_like(pre_spikes).astype(np.float32)
        else:
            self.log_changes = False
        
        for tid, is_spiked in enumerate(post_spikes[0]):
            t_ti, t_tj = self.update_spike_times(t_ti, t_tj, pre_spikes[:,tid])
        
            if is_spiked:
                presyn_spikes = np.full(pre_spikes.shape[0], False)
                weights = self.LTP(t_tj, t_ti, pre_spikes.shape[0], weights, tid)
                t_ti = 0.0
            else:
                if t_ti < self.params['tau_minus']*7:
                    weights, presyn_spikes = self.LTD(t_ti, presyn_spikes, pre_spikes[:,tid], weights, tid)
            self.mu_w_hist.append(np.mean(weights))
                    
        return weights
        
    def LTD(self, t_ti, presyn_spikes, current_spikes, w, tid):
        n_syn = current_spikes.shape[0]
        ltd = np.where(np.logical_and(current_spikes, np.logical_not(presyn_spikes)),
                       np.full(n_syn, self.params['a_minus']) * np.exp(-(t_ti/self.params['tau_minus'])),
                       np.full(n_syn, 0.0))

        new_w = np.subtract(w, ltd*self.learning_rate)    #lr: 1e-2
        
        if self.log_changes:
            self.log_delta_w[:,tid] = -ltd*self.learning_rate

        presyn_spikes = presyn_spikes | current_spikes
        return np.clip(new_w, 0.0, 1.0), presyn_spikes

    def LTP(self, t_tj, t_ti, n_syn, w, tid):
        last_spikes = np.min(t_tj, axis=0)

        ltp = np.where(last_spikes < t_ti,
                       np.full(n_syn, self.params['a_plus']) * np.exp(-(last_spikes/self.params['tau_plus'])),
                       np.full(n_syn, 0.0))

        new_w = w + ltp*self.learning_rate
        
        if self.log_changes:
            self.log_delta_w[:,tid] = ltp*self.learning_rate
        
        return np.clip(new_w, 0.0, 1.0)