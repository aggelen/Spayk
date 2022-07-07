#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:49:06 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import torchvision

class ConstantCurrentSource:
    def __init__(self, params):
        ts, te, self.t_stop, amp = params['t_cur_start'], params['t_cur_end'], params['t_stop'], params['amplitudes']
        self.dt = params['dt']
        self.steps = np.arange(int(self.t_stop / self.dt))

        # signals = np.multiply(np.ones((ts.size, self.steps.size)), amp[:, np.newaxis])
        signals = np.zeros((ts.size, self.steps.size))
        ind_s, ind_e = ts/self.dt, te/self.dt

        for i in range(ts.size):
            signals[i, int(ind_s[i]):int(ind_e[i])] = amp[i]

        self.currents = signals
        self.current_step = 0
        self.source_type = 'current'

    def plot(self):
        plt.figure()
        plt.plot(np.arange(self.currents.shape[1])*self.dt, self.currents.T)
        plt.title('Injected Currents')
        plt.xlabel('Time (ms)')
        plt.ylabel('Current ()')

    def I(self):
        I = self.currents[:,self.current_step]
        self.current_step += 1
        return I

class SpikeTrain:
    def __init__(self):
        pass

    def current_spikes(self):
        spikes = self.spikes[:,self.current_step]
        self.current_step += 1
        return spikes

    def raster_plot(self, color_array=None, first_n=None):
        f = plt.figure()
        plt.title('Raster Plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        # spike_times = []

        mean_spike_rate = np.sum(self.spikes,1).mean() / (self.t_stop/1000)
        print('Output Mean Spike Rate: {}'.format(mean_spike_rate))

        if first_n is not None:
            spike_loc = np.argwhere(self.spikes[:first_n,:])
        else:
            spike_loc = np.argwhere(self.spikes)

        sns.set()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if color_array is not None:
            # c = plt.cm.Set1(color_array[spike_loc[:,0]])
            c = []
            for k in color_array[spike_loc[:,0]]:
                c.append(colors[int(k)])
        else:
            c = colors[0]

        plt.scatter(spike_loc[:,1]*self.dt, spike_loc[:,0], s=3, color=c)


class ExternalSpikeTrain(SpikeTrain):
    def __init__(self, dt, t_stop, no_neurons, spike_train):
        super().__init__()
        self.dt = dt
        self.t_stop = t_stop
        self.no_neurons = no_neurons
        self.steps = np.arange(int(t_stop / dt))

        self.spikes = spike_train

        self.source_type = 'spike_train'
        self.current_step = 0
        
        if self.spikes.shape.__len__() == 1:
            self.mean_spike_rate = self.spikes.mean() / (self.t_stop/1000)
        else:
            self.mean_spike_rate = np.sum(self.spikes,1).mean() / (self.t_stop/1000)
        print('Spike Train Mean Spike Rate: {}'.format(self.mean_spike_rate))

    def current_spikes(self):
        if self.spikes.shape.__len__() == 1:
            self.spikes = np.expand_dims(self.spikes, axis=0)
        spikes = self.spikes[:,self.current_step]
        self.current_step += 1
        return spikes

class PoissonSpikeTrain(SpikeTrain):
    def __init__(self, dt, t_stop, no_neurons, spike_rates):
        super().__init__()
        #dt in ms
        self.dt = dt
        self.t_stop = t_stop
        self.no_neurons = no_neurons
        self.steps = np.arange(int(t_stop / dt))
        prob = np.random.uniform(0, 1, (self.steps.size, self.no_neurons))
        self.spikes = np.less(prob, np.array(spike_rates)*dt*1e-3).T
        self.source_type = 'spike_train'
        self.current_step = 0

        self.mean_spike_rate = np.sum(self.spikes,1).mean() / (self.t_stop/1000)
        print('Spike Train Mean Spike Rate: {}'.format(self.mean_spike_rate))
        
def poisson_spike_train_generator(dt, t_stop, rates):
    s = np.random.uniform(-50, 50, size=rates.size)
    spike_train = []
    for t in range(t_stop):
        prob = np.random.uniform(0, 1, rates.shape)
        spikes = np.less(prob, np.array(rates)*dt*1e-3)
        spike_train.append(spikes)
        rates = np.clip(rates + s*dt*1e-3 , 0, 90)
        ds = np.random.uniform(-5, 5, size=rates.size)
        s = np.clip(s + ds, -50, 50)
    
    # prob = np.random.uniform(0, 1, (int(t_stop / dt), rates.size))
    # spikes = np.less(prob, np.array(rates)*dt*1e-3).T
    return np.array(spike_train).T
    

class ExternalCurrentSignal:
    def __init__(self, signal):
        self.signal = signal
        self.idx = 0

    def I(self):
        I = self.signal[self.idx]
        self.idx += 1
        return I

class SpikingClassificationDataset:
    triangle = np.array([[0,0,0,0,0],
                         [0,1,0,0,0],
                         [0,1,1,0,0],
                         [0,1,1,1,0],
                         [0,0,0,0,0]])

    frame = np.array([[1,1,1,1,1],
                      [1,0,0,0,1],
                      [1,0,0,0,1],
                      [1,0,0,0,1],
                      [1,1,1,1,1]])

    square = np.array([[0,0,0,0,0],
                       [0,1,1,1,0],
                       [0,1,1,1,0],
                       [0,1,1,1,0],
                       [0,0,0,0,0]])
    
    class_templates = [triangle, frame, square]

    def __init__(self, dt, t_sample, t_stop):
        self.dt = dt
        self.t_sample = t_sample
        self.t_stop = t_stop
        self.no_samples = int(self.t_stop/self.t_sample)
        self.rate_constant = 40
        
        self.n_synapses = self.class_templates[0].flatten().size
        
        self.classes = np.random.randint(0,3,self.no_samples)
        
        target_spikes = np.zeros((self.class_templates.__len__(), t_stop))
        k = 0
        for t in range(t_stop):
            if t == int(t_sample/2) + t_sample*k:
                target_spikes[self.classes[k-1],t] = 1
                k += 1
            
        self.target_spike_train = ExternalSpikeTrain(dt, t_stop, self.class_templates.__len__(), np.array(target_spikes))
        
        
        spikes = []
        
        for c in self.classes:
            st = poisson_spike_train_generator(dt, t_sample, self.class_templates[c].flatten()*self.rate_constant)
            spikes.append(st)
            
        spikes = np.hstack(spikes)
        self.spike_train = ExternalSpikeTrain(dt, t_stop, self.class_templates[c].flatten().size, spikes)
        
class SpikingMNIST:
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(root="../datasets/", download=False)
        # img, label = self.mnist[10]
        
        self.images = []
        for i in range(10):
            indices = self.mnist.targets == i
            self.images.append(self.mnist.data[indices].detach().numpy())
            
    def get_random_sample(self, label, dt, t_stop):
        flat_arr = self.images[label][np.random.randint(self.images[label].shape[0])].flatten()
        # flat_arr = self.images[label][0].flatten()
        frates = flat_arr*(50/255) + np.random.uniform(0,1, flat_arr.shape)*15
        return poisson_spike_train_generator(dt, t_stop, frates)
            