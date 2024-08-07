import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict

#%%
class SpikeTrain:
    def __init__(self, spikes, time_params):
        self.spikes = spikes
        self.time_params = time_params      
#%%    
class PoissonSpikeTrain(SpikeTrain):
    def __init__(self, no_neurons, firing_rates, time_params):
        spikes = poisson_spike_generator(no_neurons, firing_rates, time_params)
        super().__init__(spikes, time_params)
    
def poisson_spike_generator(no_neurons, firing_rates, time_params):   
    t_start, t_stop, dt = time_params
    time_array = np.arange(t_start, t_stop, dt)
    prob = np.random.uniform(0, 1, (time_array.size, no_neurons))
    spikes = np.less(prob, np.array(firing_rates)*dt)
    return spikes.T


#%% Problem
class Problem:
	def __init__(self):
		self.runtime_path = '/home/gelenag/Dev/Spayk/new_core/first_run'
		self.neuron_dict = self.load_pickle('/home/gelenag/Dev/Spayk/new_core/first_run/neuron_dict.pickle')
		self.synapse_dict = self.load_pickle('/home/gelenag/Dev/Spayk/new_core/first_run/synapse_dict.pickle')

		self.stimuli_dict = self.load_pickle('/home/gelenag/Dev/Spayk/new_core/first_run/stimuli_dict.pickle')

		self.dt = 0.0001

		self.tsim = 1

		self.output_spikes = []

		self.channel_history = defaultdict(list)

		#%% Stimuli 
		self.stimuli = {'noiseE': PoissonSpikeTrain(1000,2.4,(0, 1, 0.0001)),
'noiseI': PoissonSpikeTrain(1000,2.4,(0, 1, 0.0001)),
'stimA': PoissonSpikeTrain(240,15,(0, 1, 0.0001)),
'stimB': PoissonSpikeTrain(240,30,(0, 1, 0.0001)),
		}
		self.current_stimuli = {'noiseE': self.stimuli['noiseE'].spikes[:,0],
'noiseI': self.stimuli['noiseI'].spikes[:,0],
'stimA': self.stimuli['stimA'].spikes[:,0],
'stimB': self.stimuli['stimB'].spikes[:,0],
'E': np.zeros(1600),
'I': np.zeros(400),
		}
		#%% neuron memb. potens.
		# >>>>>>>>> lif variables
		self.V = -55e-3*np.ones(2000)
		self.t_ref = np.zeros(2000)
		self.sAmpa_hist, self.dsAmpa_hist, self.I_syn_hist = [], [], []
		self.neuron_GLs = self.neuron_dict['GLs']
		self.neuron_VLs = self.neuron_dict['VLs']
		self.neuron_VEs = self.neuron_dict['VEs']
		self.neuron_VTs = self.neuron_dict['VTs']
		self.neuron_VRs = self.neuron_dict['VRs']
		self.neuron_CMs = self.neuron_dict['CMs']

		#%% channel states 
		self.s_AMPA_EXT__CONN_0 = np.zeros((1600,1000))
		self.s_AMPA_EXT__CONN_1 = np.zeros((400,1000))
		self.s_AMPA_EXT__CONN_2 = np.zeros((240,240))
		self.s_AMPA_EXT__CONN_3 = np.zeros((240,240))

		# derivative reset func
		self.dx_reset()
	def load_pickle(self, pkl_path):
		with open(pkl_path, 'rb') as handle:
			return pickle.load(handle)
	#%% derivative reset func
	def dx_reset(self):
		self.d_V = np.zeros(2000)
		self.d_s_AMPA_EXT__CONN_0 = np.zeros((1600,1000))
		self.d_s_AMPA_EXT__CONN_1 = np.zeros((400,1000))
		self.d_s_AMPA_EXT__CONN_2 = np.zeros((240,240))
		self.d_s_AMPA_EXT__CONN_3 = np.zeros((240,240))
	#%% euler integration func
	def integrate_synapses(self):
		#%% derivative func
		# noiseE ---AMPA_EXT---> E
		self.d_s_AMPA_EXT__CONN_0 = (-self.s_AMPA_EXT__CONN_0 / 0.002) + np.tile(self.current_stimuli['noiseE'], (1600,1)) 
		# noiseI ---AMPA_EXT---> I
		self.d_s_AMPA_EXT__CONN_1 = (-self.s_AMPA_EXT__CONN_1 / 0.002) + np.tile(self.current_stimuli['noiseI'], (400,1)) 
		# stimA ---AMPA_EXT_E---> [0:240]
		self.d_s_AMPA_EXT__CONN_2 = (-self.s_AMPA_EXT__CONN_2 / 0.002) + np.tile(self.current_stimuli['stimA'], (1600,1))[0:240,:] 
		# stimB ---AMPA_EXT_E---> [240:480]
		self.d_s_AMPA_EXT__CONN_3 = (-self.s_AMPA_EXT__CONN_3 / 0.002) + np.tile(self.current_stimuli['stimB'], (1600,1))[240:480,:] 
		#%% integrate funcs
		self.s_AMPA_EXT__CONN_0 = self.s_AMPA_EXT__CONN_0 + self.d_s_AMPA_EXT__CONN_0*self.dt
		self.s_AMPA_EXT__CONN_1 = self.s_AMPA_EXT__CONN_1 + self.d_s_AMPA_EXT__CONN_1*self.dt
		self.s_AMPA_EXT__CONN_2 = self.s_AMPA_EXT__CONN_2 + self.d_s_AMPA_EXT__CONN_2*self.dt
		self.s_AMPA_EXT__CONN_3 = self.s_AMPA_EXT__CONN_3 + self.d_s_AMPA_EXT__CONN_3*self.dt
	#%% synaptic current calculation func
	def calculate_synaptic_currents(self):
		self.I_syn = np.zeros(2000)
		self.I_syn_hist.append(self.I_syn)
		# noiseE ---AMPA_EXT---> E
		wj = self.synapse_dict['connection_list'][0]['g']*np.multiply(self.synapse_dict['connection_list'][0]['W'], self.s_AMPA_EXT__CONN_0)
		self.I_syn[0:1600] = self.I_syn[0:1600] + (self.V-self.neuron_VEs)[0:1600]*np.sum(wj, axis=1) 
		# noiseI ---AMPA_EXT---> I
		wj = self.synapse_dict['connection_list'][1]['g']*np.multiply(self.synapse_dict['connection_list'][1]['W'], self.s_AMPA_EXT__CONN_1)
		self.I_syn[1600:2000] = self.I_syn[1600:2000] + (self.V-self.neuron_VEs)[1600:2000]*np.sum(wj, axis=1) 
		# stimA ---AMPA_EXT_Subgroup_E---> [0:240]
		wj = self.synapse_dict['connection_list'][2]['g']*np.multiply(self.synapse_dict['connection_list'][2]['W'], self.s_AMPA_EXT__CONN_2)
		self.I_syn[0:1600][0:240] = self.I_syn[0:1600][0:240] + (self.V-self.neuron_VEs)[0:1600][0:240] *np.sum(wj, axis=1) 
		# stimB ---AMPA_EXT_Subgroup_E---> [240:480]
		wj = self.synapse_dict['connection_list'][3]['g']*np.multiply(self.synapse_dict['connection_list'][3]['W'], self.s_AMPA_EXT__CONN_3)
		self.I_syn[0:1600][240:480] = self.I_syn[0:1600][240:480] + (self.V-self.neuron_VEs)[0:1600][240:480] *np.sum(wj, axis=1) 
	#%% firing control
	def integrate_and_fire(self):
		is_in_rest = np.greater(self.t_ref, 0.0)
		self.t_ref = np.where(is_in_rest, self.t_ref - self.dt, self.t_ref)

		##% memb pot derivatives
		self.d_V = (-self.neuron_GLs*(self.V - self.neuron_VLs) - self.I_syn) / self.neuron_CMs
		integrated_V = self.V + self.d_V*self.dt
		self.V = np.where(np.logical_not(is_in_rest), integrated_V, self.V)
		is_fired = np.greater_equal(self.V, self.neuron_VTs)
		self.V = np.where(is_fired, self.neuron_VRs, self.V)
		self.t_ref = np.where(is_fired, self.neuron_dict['trefs'], self.t_ref)

		self.output_spikes.append(np.copy(is_fired))
	#%% data acquisition
	def data_acquisition(self, time_step):
		self.data = {}
		self.current_stimuli = {'noiseE': self.stimuli['noiseE'].spikes[:,time_step],
'noiseI': self.stimuli['noiseI'].spikes[:,time_step],
'stimA': self.stimuli['stimA'].spikes[:,time_step],
'stimB': self.stimuli['stimB'].spikes[:,time_step],
'E': np.array(self.output_spikes[-1])[0:1600],
'I': np.array(self.output_spikes[-1])[1600:2000],
		}
	#%% forward func
	def forward(self, time_idx):
		self.integrate_synapses()
		self.calculate_synaptic_currents()
		self.integrate_and_fire()
		self.data_acquisition(time_idx)
#%% Solution
problem = Problem()
time_array = np.arange(0.0, problem.tsim, problem.dt)
t_idx = 0
for t in tqdm(time_array):
	problem.forward(t_idx)
	t_idx += 1