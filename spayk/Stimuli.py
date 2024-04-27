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
from torchvision import transforms

class SimpleConstantCurrentSource:
    def __init__(self, dt, t_stop, signal):
        self.t_stop = t_stop
        self.dt = dt
        self.steps = np.arange(int(self.t_stop / self.dt))

        self.currents = signal
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
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0

    def current_spikes(self):
        spikes = self.spikes[:,self.current_step]
        self.current_step += 1
        return spikes

    def raster_plot(self, color_array=None, first_n=None, title=None, title_pad=0):
        f = plt.figure()
        if title is not None:
            plt.title(title, pad=title_pad)
        else:
            plt.title('Raster Plot', pad=title_pad)
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

        plt.scatter(spike_loc[:,1]*self.dt, spike_loc[:,0], s=2, color=c)
        plt.xlim([0,(spike_loc[:,1]*self.dt)[-1]])


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
            
        print('External Spike Train Mean Spike Rate: {}'.format(self.mean_spike_rate))

    def current_spikes(self):
        if self.spikes.shape.__len__() == 1:
            self.spikes = np.expand_dims(self.spikes, axis=0)
        spikes = self.spikes[:,self.current_step]
        self.current_step += 1
        return spikes

class SpikeInstance:
    def __init__(self, dt):
        super().__init__()
        self.dt = dt
        self.source_type = 'spike_instance'
        
    def load_spikes(self, spikes):
        self.current_spikes = spikes


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

        # self.mean_spike_rate = np.sum(self.spikes,1).mean() / (self.t_stop/1000)
        # print('Spike Train Mean Spike Rate: {}'.format(self.mean_spike_rate))
        
def poisson_spike_train_generator(dt, t_stop, rates, variable_rate=True):
    if variable_rate:
        s = np.random.uniform(-50, 50, size=rates.size)
        
    spike_train = []
    for t in range(int(t_stop/dt)):
        prob = np.random.uniform(0, 1, rates.shape)
        spikes = np.less(prob, np.array(rates)*dt*1e-3)
        spike_train.append(spikes)
        if variable_rate:
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
    def __init__(self, root='../'):
        self.transform = transforms.Compose([
           transforms.ToTensor(),
           # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist = torchvision.datasets.MNIST(root=root+"datasets/", train=True, transform=self.transform, download=True)
        # img, label = self.mnist[10]
        
        self.images = []
        for i in range(10):
            indices = self.mnist.targets == i
            self.images.append(self.mnist.data[indices].detach().numpy())
            
    def get_random_sample(self, label, dt, t_stop, jitter=10, max_fr=50):
        # flat_arr = self.images[label][np.random.randint(self.images[label].shape[0])].flatten()
        flat_arr = self.images[label][0].flatten()
        # frates = flat_arr*(50/255) + np.random.uniform(0,1, flat_arr.shape)*15
        frates = flat_arr*(max_fr/255)
        
        print('Frates mean: {}, std: {}'.format(frates.mean(), frates.std()))
        print('Frates min: {}, max: {}'.format(frates.min(), frates.max()))
        spike_train = poisson_spike_train_generator(dt, t_stop, frates)
        
        if jitter:
            #jitter | for stability 
            jitter_prob = np.random.uniform(0,1,spike_train.shape)
            jitter_st = jitter_prob < jitter*dt*1e-3
            spike_train = np.logical_or(spike_train, jitter_st)
        return spike_train, self.images[label][0]
            
    def generate_test_spike_train(self, n_samples, dt, t_sample):
        self.mnist = torchvision.datasets.MNIST(root="../datasets/", train=False, transform=self.transform, download=True)
        spikes = []
        labels = []
        for i in range(n_samples):
            img, label = self.mnist[i]
            flat_arr = img.detach().numpy().flatten()
            frates = flat_arr*(50/255)
            spike_train = poisson_spike_train_generator(dt, t_sample, frates)
            spikes.append(spike_train)
            labels.append(label)
        
        spikes = np.hstack(spikes)
        
        jitter_prob = np.random.uniform(0,1,spikes.shape)
        jitter = jitter_prob < 10*dt*1e-3
        spikes = np.logical_or(spikes, jitter)
        
        return spikes, np.array(labels)
    
    def special_spike_train(self, dt, t_stop, t_sample, distribute_5=True):
        self.mnist = torchvision.datasets.MNIST(root="../datasets/", train=True, transform=self.transform, download=True)
        #%% base spike train
        no_neurons = 784*2
        
        print('creating stimuli')
        r = np.random.uniform(0, 30, size=(no_neurons))
        s = np.random.uniform(-50, 50, size=(no_neurons))
        spike_train = []
        for t in range(t_stop):
            prob = np.random.uniform(0, 1, r.shape)
            spikes = np.less(prob, np.array(r)*dt*1e-3)
            spike_train.append(spikes)
            r = np.clip(r + s*dt*1e-3 , 0, 30)
            ds = np.random.uniform(-5, 5, size=(no_neurons))
            s = np.clip(s + ds, -50, 50)

        spike_train = np.array(spike_train).T
        
        selected_labels = np.random.choice(np.arange(10), int(t_stop/t_sample))
        
        for sid, sl in enumerate(selected_labels):
            sel_imgs = self.images[sl]
            flat_arr = sel_imgs[np.random.randint(sel_imgs.shape[0])].flatten()
            frates = flat_arr*(50/255)
            pattern = poisson_spike_train_generator(dt, t_sample, frates)
            
            jitter_prob = np.random.uniform(0,1,pattern.shape)
            jitter = jitter_prob < 10*dt*1e-3
            pattern = np.logical_or(pattern, jitter)
        
            spike_train[:784, sid*50:(sid+1)*50] = np.copy(pattern)
        
        if distribute_5:
            pattern_img = self.images[5][0]
            flat_arr = pattern_img.flatten()
            frates = flat_arr*(50/255)
            pattern = poisson_spike_train_generator(dt, t_sample, frates)
            jitter_prob = np.random.uniform(0,1,pattern.shape)
            jitter = jitter_prob < 10*dt*1e-3
            pattern = np.logical_or(pattern, jitter)
            
            first_repeat = 50
            spike_train[:784, first_repeat:first_repeat+50] = np.copy(pattern)
    
            repeat_times = [first_repeat]
            repeat_mode = [1]
            last_repeat_time = first_repeat
    
            for t in range(t_stop-50):
                if t > last_repeat_time + 100:
                    r = np.random.uniform(0,1,25)
                    mask = r < 0.25
                    c = np.argwhere(mask).min()
                    start_at = t + c*50
                    if start_at + 50 > t_stop:
                        break
    
                    repeat_times.append(start_at)
                    last_repeat_time = start_at
                
                    spike_train[:784, last_repeat_time:last_repeat_time+50] = np.copy(pattern)
                    repeat_mode.append(1)

            repeat_times = np.array(repeat_times)
        # repeat_labels = np.array(repeat_labels)
        else:
            repeat_times = False
        
        return spike_train, repeat_times, selected_labels
    
    def special_spike_train2(self, dt, t_stop, t_sample, distribute_5=True):
        self.mnist = torchvision.datasets.MNIST(root="../datasets/", train=True, transform=self.transform, download=True)
        #%% base spike train
        no_neurons = 784*2
        max_fr = 60
        r = np.random.uniform(0, max_fr, size=(no_neurons))
        s = np.random.uniform(-50, 50, size=(no_neurons))
        spike_train = []
        for t in range(t_stop):
            prob = np.random.uniform(0, 1, r.shape)
            spikes = np.less(prob, np.array(r)*dt*1e-3)
            spike_train.append(spikes)
            r = np.clip(r + s*dt*1e-3 , 0, max_fr)
            ds = np.random.uniform(-5, 5, size=(no_neurons))
            s = np.clip(s + ds, -50, 50)

        spike_train = np.array(spike_train).T
        
        selected_labels = np.random.choice(np.arange(10), int(t_stop/t_sample))
        
        for sid, sl in enumerate(selected_labels):
            sel_imgs = self.images[sl]
            flat_arr = sel_imgs[np.random.randint(sel_imgs.shape[0])].flatten()
            frates = flat_arr*(90/255)
            pattern = poisson_spike_train_generator(dt, t_sample, frates)
            
            jitter_prob = np.random.uniform(0,1,pattern.shape)
            jitter = jitter_prob < 10*dt*1e-3
            pattern = np.logical_or(pattern, jitter)
        
            spike_train[:784, sid*50:(sid+1)*50] = np.copy(pattern)
        
        if distribute_5:
            pattern_img = self.images[5][0]
            flat_arr = pattern_img.flatten()
            frates = flat_arr*(50/255)
            pattern = poisson_spike_train_generator(dt, t_sample, frates)
            jitter_prob = np.random.uniform(0,1,pattern.shape)
            jitter = jitter_prob < 10*dt*1e-3
            pattern = np.logical_or(pattern, jitter)
            
            first_repeat = 50
            spike_train[:784, first_repeat:first_repeat+50] = np.copy(pattern)
    
            repeat_times = [first_repeat]
            repeat_mode = [1]
            last_repeat_time = first_repeat
    
            for t in range(t_stop-50):
                if t > last_repeat_time + 100:
                    r = np.random.uniform(0,1,25)
                    mask = r < 0.25
                    c = np.argwhere(mask).min()
                    start_at = t + c*50
                    if start_at + 50 > t_stop:
                        break
    
                    repeat_times.append(start_at)
                    last_repeat_time = start_at
                
                    spike_train[:784, last_repeat_time:last_repeat_time+50] = np.copy(pattern)
                    repeat_mode.append(1)

            repeat_times = np.array(repeat_times)
        # repeat_labels = np.array(repeat_labels)
        else:
            repeat_times = False
        
        return spike_train, repeat_times, selected_labels
                
        
            
        