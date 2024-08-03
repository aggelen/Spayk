import numpy as np
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

		self.output_spikes = []

		self.channel_history = defaultdict(list)

		#%% Stimuli 
		self.stimuli = {'external_stim': PoissonSpikeTrain(25,25,(0, 1, 0.0001)),
		}
		self.current_stimuli = {'external_stim': self.stimuli['external_stim'].spikes[:,0],
'N': np.zeros(2),
'E': np.zeros(2),
'I': np.zeros(2),
		}
		#%% neuron memb. potens.
		# >>>>>>>>> lif variables
		self.V = -55e-3*np.ones(6)
		self.t_ref = np.zeros(6)
		self.sAmpa_hist, self.dsAmpa_hist, self.I_syn_hist = [], [], []
		self.neuron_GLs = self.neuron_dict['GLs']
		self.neuron_VLs = self.neuron_dict['VLs']
		self.neuron_VEs = self.neuron_dict['VEs']
		self.neuron_VTs = self.neuron_dict['VTs']
		self.neuron_VRs = self.neuron_dict['VRs']
		self.neuron_CMs = self.neuron_dict['CMs']

		#%% channel states 
		self.s_AMPA_EXT__CONN_0 = np.zeros((2,25))
		self.s_AMPA__CONN_1 = np.zeros((2,2))
		self.s_NMDA__CONN_2 = np.zeros((2,2))
		self.x_NMDA__CONN_2 = np.zeros((2,2))
		self.s_AMPA__CONN_3 = np.zeros((2,2))
		self.s_NMDA__CONN_4 = np.zeros((2,2))
		self.x_NMDA__CONN_4 = np.zeros((2,2))
		self.s_AMPA__CONN_5 = np.zeros((2,2))
		self.s_NMDA__CONN_6 = np.zeros((2,2))
		self.x_NMDA__CONN_6 = np.zeros((2,2))
		self.s_GABA__CONN_7 = np.zeros((2,2))

		# derivative reset func
		self.dx_reset()
	def load_pickle(self, pkl_path):
		with open(pkl_path, 'rb') as handle:
			return pickle.load(handle)
	#%% derivative reset func
	def dx_reset(self):
		self.d_V = np.zeros(6)
		self.d_s_AMPA_EXT__CONN_0 = np.zeros((2,25))
		self.d_s_AMPA__CONN_1 = np.zeros((2,2))
		self.d_s_NMDA__CONN_2 = np.zeros((2,2))
		self.d_x_NMDA__CONN_2 = np.zeros((2,2))
		self.d_s_AMPA__CONN_3 = np.zeros((2,2))
		self.d_s_NMDA__CONN_4 = np.zeros((2,2))
		self.d_x_NMDA__CONN_4 = np.zeros((2,2))
		self.d_s_AMPA__CONN_5 = np.zeros((2,2))
		self.d_s_NMDA__CONN_6 = np.zeros((2,2))
		self.d_x_NMDA__CONN_6 = np.zeros((2,2))
		self.d_s_GABA__CONN_7 = np.zeros((2,2))
	#%% euler integration func
	def integrate_synapses(self):
		#%% derivative func
		# external_stim ---AMPA_EXT---> N
		self.d_s_AMPA_EXT__CONN_0 = (-self.s_AMPA_EXT__CONN_0 / 0.002) + np.tile(self.current_stimuli['external_stim'], (2,1)) 
		# N ---AMPA---> E
		self.d_s_AMPA__CONN_1 = (-self.s_AMPA__CONN_1 / 0.002) + np.tile(self.current_stimuli['N'], (2,1)) 
		# N ---NMDA---> E
		self.d_x_NMDA__CONN_2 = (-self.x_NMDA__CONN_2 / 0.002) + np.tile(self.current_stimuli['N'], (2,1)) 
		self.d_s_NMDA__CONN_2 = (-self.s_NMDA__CONN_2 / 0.1) + 500.0*self.x_NMDA__CONN_2*(1 - self.s_NMDA__CONN_2) 
		# N ---AMPA---> I
		self.d_s_AMPA__CONN_3 = (-self.s_AMPA__CONN_3 / 0.002) + np.tile(self.current_stimuli['N'], (2,1)) 
		# N ---NMDA---> I
		self.d_x_NMDA__CONN_4 = (-self.x_NMDA__CONN_4 / 0.002) + np.tile(self.current_stimuli['N'], (2,1)) 
		self.d_s_NMDA__CONN_4 = (-self.s_NMDA__CONN_4 / 0.1) + 500.0*self.x_NMDA__CONN_4*(1 - self.s_NMDA__CONN_4) 
		# N ---AMPA---> I
		self.d_s_AMPA__CONN_5 = (-self.s_AMPA__CONN_5 / 0.002) + np.tile(self.current_stimuli['N'], (2,1)) 
		# N ---NMDA---> I
		self.d_x_NMDA__CONN_6 = (-self.x_NMDA__CONN_6 / 0.002) + np.tile(self.current_stimuli['N'], (2,1)) 
		self.d_s_NMDA__CONN_6 = (-self.s_NMDA__CONN_6 / 0.1) + 500.0*self.x_NMDA__CONN_6*(1 - self.s_NMDA__CONN_6) 
		# I ---GABA---> E
		self.d_s_GABA__CONN_7 = (-self.s_GABA__CONN_7 / 0.005) + np.tile(self.current_stimuli['I'], (2,1)) 
		#%% integrate funcs
		self.s_AMPA_EXT__CONN_0 = self.s_AMPA_EXT__CONN_0 + self.d_s_AMPA_EXT__CONN_0*self.dt
		self.channel_history['s_AMPA_EXT__CONN_0'].append(self.s_AMPA_EXT__CONN_0)
		self.s_AMPA__CONN_1 = self.s_AMPA__CONN_1 + self.d_s_AMPA__CONN_1*self.dt
		self.channel_history['s_AMPA__CONN_1'].append(self.s_AMPA__CONN_1)
		self.s_NMDA__CONN_2 = self.s_NMDA__CONN_2 + self.d_s_NMDA__CONN_2*self.dt
		self.x_NMDA__CONN_2 = self.x_NMDA__CONN_2 + self.d_x_NMDA__CONN_2*self.dt
		self.channel_history['s_NMDA__CONN_2'].append(self.s_NMDA__CONN_2)
		self.s_AMPA__CONN_3 = self.s_AMPA__CONN_3 + self.d_s_AMPA__CONN_3*self.dt
		self.channel_history['s_AMPA__CONN_3'].append(self.s_AMPA__CONN_3)
		self.s_NMDA__CONN_4 = self.s_NMDA__CONN_4 + self.d_s_NMDA__CONN_4*self.dt
		self.x_NMDA__CONN_4 = self.x_NMDA__CONN_4 + self.d_x_NMDA__CONN_4*self.dt
		self.channel_history['s_NMDA__CONN_4'].append(self.s_NMDA__CONN_4)
		self.s_AMPA__CONN_5 = self.s_AMPA__CONN_5 + self.d_s_AMPA__CONN_5*self.dt
		self.channel_history['s_AMPA__CONN_5'].append(self.s_AMPA__CONN_5)
		self.s_NMDA__CONN_6 = self.s_NMDA__CONN_6 + self.d_s_NMDA__CONN_6*self.dt
		self.x_NMDA__CONN_6 = self.x_NMDA__CONN_6 + self.d_x_NMDA__CONN_6*self.dt
		self.channel_history['s_NMDA__CONN_6'].append(self.s_NMDA__CONN_6)
		self.s_GABA__CONN_7 = self.s_GABA__CONN_7 + self.d_s_GABA__CONN_7*self.dt
		self.channel_history['s_GABA__CONN_7'].append(self.s_GABA__CONN_7)
	#%% synaptic current calculation func
	def calculate_synaptic_currents(self):
		self.I_syn = np.zeros(6)
		self.I_syn_hist.append(self.I_syn)
		# external_stim ---AMPA_EXT---> N
		wj = self.synapse_dict['connection_list'][0]['g']*np.multiply(self.synapse_dict['connection_list'][0]['W'], self.s_AMPA_EXT__CONN_0)
		self.I_syn[0:2] = self.I_syn[0:2] + (self.V-self.neuron_VEs)[0:2]*np.sum(wj, axis=1) 
		# N ---AMPA---> E
		wj = self.synapse_dict['connection_list'][1]['g']*np.multiply(self.synapse_dict['connection_list'][1]['W'], self.s_AMPA__CONN_1)
		self.I_syn[2:4] = self.I_syn[2:4] + (self.V-self.neuron_VEs)[2:4]*np.sum(wj, axis=1) 
		# N ---NMDA---> E
		wj = self.synapse_dict['connection_list'][2]['g']*np.multiply(self.synapse_dict['connection_list'][2]['W'], self.s_NMDA__CONN_2)
		self.I_syn[2:4] = self.I_syn[2:4] + (self.V-self.neuron_VEs)[2:4]*(np.sum(wj, axis=1))/(1 + (0.001*np.exp(-0.062*self.V[2:4])/3.57)) 
		# N ---AMPA---> I
		wj = self.synapse_dict['connection_list'][3]['g']*np.multiply(self.synapse_dict['connection_list'][3]['W'], self.s_AMPA__CONN_3)
		self.I_syn[4:6] = self.I_syn[4:6] + (self.V-self.neuron_VEs)[4:6]*np.sum(wj, axis=1) 
		# N ---NMDA---> I
		wj = self.synapse_dict['connection_list'][4]['g']*np.multiply(self.synapse_dict['connection_list'][4]['W'], self.s_NMDA__CONN_4)
		self.I_syn[4:6] = self.I_syn[4:6] + (self.V-self.neuron_VEs)[4:6]*(np.sum(wj, axis=1))/(1 + (0.001*np.exp(-0.062*self.V[4:6])/3.57)) 
		# N ---AMPA---> I
		wj = self.synapse_dict['connection_list'][5]['g']*np.multiply(self.synapse_dict['connection_list'][5]['W'], self.s_AMPA__CONN_5)
		self.I_syn[4:6] = self.I_syn[4:6] + (self.V-self.neuron_VEs)[4:6]*np.sum(wj, axis=1) 
		# N ---NMDA---> I
		wj = self.synapse_dict['connection_list'][6]['g']*np.multiply(self.synapse_dict['connection_list'][6]['W'], self.s_NMDA__CONN_6)
		self.I_syn[4:6] = self.I_syn[4:6] + (self.V-self.neuron_VEs)[4:6]*(np.sum(wj, axis=1))/(1 + (0.001*np.exp(-0.062*self.V[4:6])/3.57)) 
		# I ---GABA---> E
		wj = self.synapse_dict['connection_list'][7]['g']*np.multiply(self.synapse_dict['connection_list'][7]['W'], self.s_GABA__CONN_7)
		self.I_syn[2:4] = self.I_syn[2:4] + (self.V+70e-3)[2:4]*np.sum(wj, axis=1) 
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
		self.current_stimuli = {'external_stim': self.stimuli['external_stim'].spikes[:,time_step],
'N': np.array(self.output_spikes[-1])[0:2],
'E': np.array(self.output_spikes[-1])[2:4],
'I': np.array(self.output_spikes[-1])[4:6],
		}
	#%% forward func
	def forward(self, time_idx):
		self.integrate_synapses()
		self.calculate_synaptic_currents()
		self.integrate_and_fire()
		self.data_acquisition(time_idx)
#%% Solution
problem = Problem()
