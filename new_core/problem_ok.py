import numpy as np
from tqdm import tqdm
from spayk.Stimuli import *
import pickle
from collections import defaultdict

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
'stimA': PoissonSpikeTrain(240,5,(0, 1, 0.0001)),
'stimB': PoissonSpikeTrain(240,10,(0, 1, 0.0001)),
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
	#%% euler integration func
	def integrate_synapses(self):
		#%% derivative func
		# noiseE ---AMPA_EXT---> E
		self.d_s_AMPA_EXT__CONN_0 = (-self.s_AMPA_EXT__CONN_0 / 0.002) 
		# noiseI ---AMPA_EXT---> I
		self.d_s_AMPA_EXT__CONN_1 = (-self.s_AMPA_EXT__CONN_1 / 0.002) 
		#%% integrate funcs
		self.s_AMPA_EXT__CONN_0 = self.s_AMPA_EXT__CONN_0 + self.d_s_AMPA_EXT__CONN_0*self.dt
		prob = np.random.uniform(0, 1, (1600, self.stimuli_dict['noiseE'].no_neurons))
		spikes = np.less(prob, np.array(self.stimuli_dict['noiseE'].firing_rates)*self.dt)
		self.s_AMPA_EXT__CONN_0 = self.s_AMPA_EXT__CONN_0 + spikes 
		self.s_AMPA_EXT__CONN_1 = self.s_AMPA_EXT__CONN_1 + self.d_s_AMPA_EXT__CONN_1*self.dt
		prob = np.random.uniform(0, 1, (400, self.stimuli_dict['noiseI'].no_neurons))
		spikes = np.less(prob, np.array(self.stimuli_dict['noiseI'].firing_rates)*self.dt)
		self.s_AMPA_EXT__CONN_1 = self.s_AMPA_EXT__CONN_1 + spikes 
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
	#%% firing control
	def integrate_and_fire(self):
		is_in_rest = np.greater(self.t_ref, 0.0)
		self.t_ref = np.where(is_in_rest, self.t_ref - self.dt, np.copy(self.t_ref))

		##% memb pot derivatives
		d_V = (-self.neuron_GLs*(self.V - self.neuron_VLs) - self.I_syn) / self.neuron_CMs
		self.d_V = np.where(is_in_rest, np.zeros(2000), d_V)
		integrated_V = self.V + self.d_V*self.dt
		self.V = np.where(np.logical_not(is_in_rest), integrated_V, np.copy(self.V))
		is_fired = np.greater_equal(self.V, self.neuron_VTs)
		self.V = np.where(is_fired, self.neuron_VRs, np.copy(self.V))
		self.t_ref = np.where(is_fired, self.neuron_dict['trefs'], np.copy(self.t_ref))

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