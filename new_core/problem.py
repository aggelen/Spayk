import numpy as np
from spayk.Stimuli import *
import pickle

#%% Problem
class Problem:
	def __init__(self):
		self.runtime_path = '/home/gelenag/Dev/Spayk/new_core/first_run'
		self.neuron_dict = self.load_pickle('/home/gelenag/Dev/Spayk/new_core/first_run/neuron_dict.pickle')
		self.synapse_dict = self.load_pickle('/home/gelenag/Dev/Spayk/new_core/first_run/synapse_dict.pickle')

		self.stimuli_dict = self.load_pickle('/home/gelenag/Dev/Spayk/new_core/first_run/stimuli_dict.pickle')

		self.dt = 0.0001

		self.output_spikes = []

		#%% Stimuli 
		self.stimuli = {'external_stim': PoissonSpikeTrain(25,25,(0, 1, 0.0001)),
		}
		self.current_stimuli = {'external_stim': self.stimuli['external_stim'].spikes[:,0],
'N': np.zeros(1),
		}
		#%% neuron memb. potens.
		# >>>>>>>>> lif variables
		self.V = -55e-3*np.ones(1)
		self.t_ref = np.zeros(1)
		self.sAmpa_hist, self.dsAmpa_hist, self.I_syn_hist = [], [], []
		self.neuron_GLs = self.neuron_dict['GLs']
		self.neuron_VLs = self.neuron_dict['VLs']
		self.neuron_VEs = self.neuron_dict['VEs']
		self.neuron_VTs = self.neuron_dict['VTs']
		self.neuron_VRs = self.neuron_dict['VRs']
		self.neuron_CMs = self.neuron_dict['CMs']

		#%% channel states 
		self.s_AMPA_EXT__CONN_0 = np.zeros((1,25))

		# derivative reset func
		self.dx_reset()
	def load_pickle(self, pkl_path):
		with open(pkl_path, 'rb') as handle:
			return pickle.load(handle)
	#%% derivative reset func
	def dx_reset(self):
		self.d_V = np.zeros(1)
		self.d_s_AMPA_EXT__CONN_0 = np.zeros((1,25))
	#%% euler integration func
	def integrate_synapses(self):
		#%% derivative func
		# external_stim ---AMPA_EXT---> N
		self.d_s_AMPA_EXT__CONN_0 = (-self.s_AMPA_EXT__CONN_0 / 0.002) + np.tile(self.current_stimuli['external_stim'], (1,1)) 
		self.dsAmpa_hist.append(self.d_s_AMPA_EXT__CONN_0)
		#%% integrate funcs
		self.s_AMPA_EXT__CONN_0 = self.s_AMPA_EXT__CONN_0 + self.d_s_AMPA_EXT__CONN_0*self.dt
		self.sAmpa_hist.append(self.s_AMPA_EXT__CONN_0)
	#%% synaptic current calculation func
	def calculate_synaptic_currents(self):
		self.I_syn = np.zeros(1)
		# external_stim ---AMPA_EXT---> N
		wj = self.synapse_dict['connection_list'][0]['g']*np.multiply(self.synapse_dict['connection_list'][0]['W'], self.s_AMPA_EXT__CONN_0)
		self.I_syn[0:1] = self.I_syn[0:1] + (self.V-self.neuron_VEs)[0:1]*np.sum(wj, axis=1) 
		self.I_syn_hist.append(np.copy(self.I_syn))
	#%% firing control
	def integrate_and_fire(self):
		is_in_rest = np.greater(self.t_ref, 0.0)[0]
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
		self.current_stimuli = {'external_stim': self.stimuli['external_stim'].spikes[:,time_step],
		}
	#%% forward func
	def forward(self, time_idx):
		self.integrate_synapses()
		self.calculate_synaptic_currents()
		self.integrate_and_fire()
		self.data_acquisition(time_idx)
#%% Solution
problem = Problem()
