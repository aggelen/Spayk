import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

#%% Problem class
class Problem:
    def __init__(self):
        #%% Class params
        self.dt = 0.1e-3
        self.tsim = 1
        self.output_spikes = np.empty((int(np.ceil(self.tsim/self.dt)),1))
        self.stim = None
        #%% >>>>>>>>> lif variables
        self.V = -57e-3*np.ones(1)
        self.t_ref = np.zeros(1)
        self.I_syn = np.zeros(1)
        self.last_spikes = np.zeros(1)
    
        self.log = defaultdict(list)

        self.last_spikes_delayed = np.zeros(2000)

        #%% channel states 
        self.VE = 0

        self.VR = -0.055
        self.VT = -0.05

        self.GL = 2.5e-08
        self.TREF = 0.002
        self.CM = 5e-10

        self.runtime_path = '/home/gelenag/Dev/Spayk/new_core/first_run'
        self.gW = self.load_pickle('{}/gW_dict.pickle'.format(self.runtime_path))

    def load_pickle(self, pkl_path):
        with open(pkl_path, 'rb') as handle:
            return pickle.load(handle)
        
    #%% firing control
    def integrate_and_fire(self, time_idx):
       self.t_ref = np.maximum(self.t_ref - self.dt, 0)
       non_refractory = self.t_ref == 0
       
       self.V[non_refractory] += self.dt * (-self.GL*(self.V - (-70e-3)) - self.I_syn) / self.CM
       
       is_fired = self.V >= self.VT
       
       self.t_ref[is_fired] = self.TREF
       
       # Reset membrane potentials for neurons that spiked
       self.V[is_fired] = self.VR
       
       self.output_spikes[time_idx] = np.copy(is_fired)
    
            
    def integrate_and_fire(self, time_idx):
        is_in_rest = np.greater_equal(self.t_ref, 0.0)
        self.t_ref = np.where(is_in_rest, self.t_ref - self.dt, self.t_ref)

        ##% memb pot derivatives
        d_V = (-self.GL*(self.V - (-70e-3)) - self.I_syn) / self.CM
        d_V = np.where(is_in_rest, np.zeros(1), d_V)
        self.V = self.V + d_V*self.dt
        is_fired = np.greater_equal(self.V, self.VT)
        if np.sum(is_fired):
            self.V = self.VR
            self.t_ref= self.TREF

        self.output_spikes[time_idx] = np.copy(is_fired)
        if time_idx > 5:
            self.last_spikes_delayed = self.output_spikes[time_idx-5]
        else:
            self.last_spikes_delayed = np.zeros(1)
            
    #%% forward func
    def forward(self, time_idx):
        self.I_syn = -1e-9
        self.integrate_and_fire(time_idx)

#%% Solution
problem = Problem()
time_array = np.arange(0.0, problem.tsim, problem.dt)
t_idx = 0
for t in tqdm(time_array):
    problem.forward(t_idx)
    t_idx += 1
    
#%% Output Spikes
import matplotlib.pyplot as plt
op_spikes = np.array(problem.output_spikes).T
# spike_loc_A = np.argwhere(op_spikes)
# plt.plot(spike_loc_A[:,1], spike_loc_A[:,0], '.', markersize=2, color='darkred')

print('FR: {}'.format(op_spikes.sum()/problem.tsim))
    
