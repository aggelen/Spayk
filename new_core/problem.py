import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

#%% Stimuli class
class Stimuli:
	def __init__(self):
		self.dt = 0.0001
		self.query_id = 0
		self.runtime_path = '/home/gelenag/Dev/Spayk/new_core/first_run'
		self.stim_dict = self.load_pickle('{}/stimuli_dict.pickle'.format(self.runtime_path))


	def poisson_generator_CONN_0(self):
		prob = np.random.uniform(0, 1, (1600, 1000))
		return np.less(prob, np.array([2.4])*self.dt)

	def poisson_generator_CONN_1(self):
		prob = np.random.uniform(0, 1, (400, 1000))
		return np.less(prob, np.array([2.4])*self.dt)

	def poisson_generator_CONN_2(self):
		prob = np.random.uniform(0, 1, (240, 240))
		return np.less(prob, self.stim_dict['stimA'].firing_rates[self.query_id]*self.dt)

	def poisson_generator_CONN_3(self):
		prob = np.random.uniform(0, 1, (240, 240))
		return np.less(prob, self.stim_dict['stimB'].firing_rates[self.query_id]*self.dt)
	def load_pickle(self, pkl_path):
		with open(pkl_path, 'rb') as handle:
			return pickle.load(handle)

	def step(self):
		self.query_id = self.query_id + 1

#%% Problem class
class Problem:
	def __init__(self):
		#%% Class params
		self.dt = 0.0001
		self.tsim = 4
		self.output_spikes = np.empty((40000,2000))
		self.stim = None
		#%% >>>>>>>>> lif variables
		self.V = -53e-3*np.ones(2000)
		self.t_ref = np.zeros(2000)
		self.I_syn = np.zeros(2000)
		self.last_spikes = np.zeros(2000)
		self.s_AMPA_ext = np.zeros(2000).astype(np.float64)
		self.s_AMPA = np.zeros(2000).astype(np.float64)
		self.s_GABA = np.zeros(2000)
		self.x_NMDA = np.zeros(2000)
		self.s_NMDA = np.zeros(2000)
		self.ds_NMDA = np.zeros(2000)
		self.log = defaultdict(list)
		self.wj_NMDA = np.zeros(2000)
		self.last_spikes_delayed = np.zeros(2000)
		self.x_NMDA_hist = []  
		self.s_NMDA_hist = []  
		self.I_AMPA_hist = []  
		#%% channel states 
		self.VE = np.hstack([np.full(1600, 0), 
np.full(400, 0)])
		self.VI = np.hstack([np.full(1600, -0.07), 
np.full(400, -0.07)])
		self.VR = np.hstack([np.full(1600, -0.055), 
np.full(400, -0.055)])
		self.VT = np.hstack([np.full(1600, -0.05), 
np.full(400, -0.05)])
		self.GL = np.hstack([np.full(1600, 2.5e-08), 
np.full(400, 2e-08)])
		self.TREF = np.hstack([np.full(1600, 0.002), 
np.full(400, 0.001)])
		self.CM = np.hstack([np.full(1600, 5e-10), 
np.full(400, 2e-10)])
		self.runtime_path = '/home/gelenag/Dev/Spayk/new_core/first_run'
		self.gW = self.load_pickle('{}/gW_dict.pickle'.format(self.runtime_path))

	def load_pickle(self, pkl_path):
		with open(pkl_path, 'rb') as handle:
			return pickle.load(handle)

	def integrate_CONN_0(self):
		# noiseE ---AMPA_EXT---> E
		self.s_AMPA_ext[0:1600] += np.einsum('ij,ij->i', self.stim.poisson_generator_CONN_0(), self.gW[0][1]*self.gW[0][0]) 
	def integrate_CONN_1(self):
		# noiseI ---AMPA_EXT---> I
		self.s_AMPA_ext[1600:2000] += np.einsum('ij,ij->i', self.stim.poisson_generator_CONN_1(), self.gW[1][1]*self.gW[1][0]) 
	def integrate_CONN_2(self):
		# stimA ---AMPA_EXT_Subgroup_E---> [0:240]
		self.s_AMPA_ext[0:1600][0:240] += np.einsum('ij,ij->i', self.stim.poisson_generator_CONN_2(), self.gW[2][1]*self.gW[2][0]) 
	def integrate_CONN_3(self):
		# stimB ---AMPA_EXT_Subgroup_E---> [240:480]
		self.s_AMPA_ext[0:1600][240:480] += np.einsum('ij,ij->i', self.stim.poisson_generator_CONN_3(), self.gW[3][1]*self.gW[3][0]) 
	def integrate_CONN_4(self):
		# E ---AMPA---> E
		self.s_AMPA[0:1600] = self.s_AMPA[0:1600] + np.einsum('ij,j->i', (self.gW[4][1]*self.gW[4][0]), self.last_spikes_delayed[0:1600]) 
	def integrate_CONN_5(self):
		# E ---NMDA---> E
		self.s_NMDA_total[0:1600] = self.gW[5][0]*np.einsum('j,ij->i', self.s_NMDA[0:1600], self.gW[5][1])
	def integrate_CONN_6(self):
		# E ---AMPA---> I
		self.s_AMPA[1600:2000] = self.s_AMPA[1600:2000] + np.einsum('ij,j->i', (self.gW[6][1]*self.gW[6][0]), self.last_spikes_delayed[0:1600]) 
	def integrate_CONN_7(self):
		# E ---NMDA---> I
		self.s_NMDA_total[1600:2000] = self.gW[7][0]*np.einsum('j,ij->i', self.s_NMDA[0:1600], self.gW[7][1])
	def integrate_CONN_8(self):
		# I ---AMPA---> E
		self.s_GABA[0:1600] += np.einsum('ij,j->i', self.gW[8][1]*self.gW[8][0], self.last_spikes_delayed[1600:2000]) 
	def integrate_CONN_9(self):
		# I ---AMPA---> I
		self.s_GABA[1600:2000] += np.einsum('ij,j->i', self.gW[9][1]*self.gW[9][0], self.last_spikes_delayed[1600:2000]) 
	#%% firing control
	def integrate_and_fire(self, time_idx):
		self.t_ref = np.maximum(self.t_ref - self.dt, 0) 
		non_refractory = self.t_ref == 0 
		self.V[non_refractory] += self.dt * (-self.GL[non_refractory]*(self.V[non_refractory]  - (-70e-3)) - self.I_syn[non_refractory]) / self.CM[non_refractory]  
		is_fired = self.V >= self.VT 
		self.t_ref[is_fired] = self.TREF[is_fired] 
		self.V[is_fired] = self.VR[is_fired] 
		self.output_spikes[time_idx] = np.copy(is_fired)
		if time_idx > 5:
			self.last_spikes_delayed = self.output_spikes[time_idx-5]
		else:
			self.last_spikes_delayed = np.zeros(2000)
	#%% forward func
	def forward(self, time_idx):
		self.I_syn = np.zeros(2000)
		self.s_NMDA_total = np.zeros(2000)
		self.s_AMPA_ext = self.s_AMPA_ext + (-self.s_AMPA_ext / 0.002)*self.dt  
		self.s_AMPA = self.s_AMPA + (-self.s_AMPA / 0.002)*self.dt  
		self.s_GABA = self.s_GABA + (-self.s_GABA / 0.005)*self.dt  
		self.x_NMDA = self.x_NMDA + (-self.x_NMDA / 0.002)*self.dt  
		self.x_NMDA += self.last_spikes_delayed 
		self.s_NMDA = self.s_NMDA + ((-self.s_NMDA / 0.1) + 500.0*self.x_NMDA*(1-self.s_NMDA))*self.dt 
		self.integrate_CONN_0()
		self.integrate_CONN_1()
		self.integrate_CONN_2()
		self.integrate_CONN_3()
		self.integrate_CONN_4()
		self.integrate_CONN_5()
		self.integrate_CONN_6()
		self.integrate_CONN_7()
		self.integrate_CONN_8()
		self.integrate_CONN_9()
		self.I_syn = np.multiply((self.V-self.VE),(self.s_AMPA_ext + self.s_AMPA)) 
		self.I_syn = self.I_syn + np.multiply((self.V-self.VI),self.s_GABA) 
		self.I_NMDA = self.s_NMDA_total * (self.V - self.VE) / (1 + np.exp(-0.062 * self.V/1e-3) * (1 / 3.57)) 
		self.I_syn = self.I_syn + self.I_NMDA 
		self.log['x_NMDA'].append(np.copy(self.x_NMDA[0]))  
		self.log['I_NMDA'].append(np.copy(self.I_NMDA[0]))  
		self.log['s_NMDA_total'].append(np.copy(self.s_NMDA_total[0]))  
		self.integrate_and_fire(time_idx)
		self.stim.step()

#%% Solution
problem = Problem()
problem.stim = Stimuli()
time_array = np.arange(0.0, problem.tsim, problem.dt)
t_idx = 0
for t in tqdm(time_array):
	problem.forward(t_idx)
	t_idx += 1