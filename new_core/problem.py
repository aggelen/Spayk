import pickle
import numpy as np
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
		self.tsim = 0.2
		self.output_spikes = np.empty((2000,2000))
		self.stim = None
		#%% >>>>>>>>> lif variables
		self.V = -55e-3*np.ones(2000)
		self.t_ref = np.zeros(2000)
		self.I_syn = np.zeros(2000)
		self.last_spikes = np.zeros(2000)
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
		self.s_AMPA_EXT__CONN_0 = np.zeros((1600,1000))
		self.s_AMPA_EXT__CONN_1 = np.zeros((400,1000))
		self.s_AMPA_EXT__CONN_2 = np.zeros((240,240))
		self.s_AMPA_EXT__CONN_3 = np.zeros((240,240))
		self.s_AMPA__CONN_4 = np.zeros((1600,1600))
		self.s_AMPA__CONN_5 = np.zeros((400,1600))
		self.runtime_path = '/home/gelenag/Dev/Spayk/new_core/first_run'
		self.gW = self.load_pickle('{}/gW_dict.pickle'.format(self.runtime_path))

	def load_pickle(self, pkl_path):
		with open(pkl_path, 'rb') as handle:
			return pickle.load(handle)

	def integrate_CONN_0(self):
		# noiseE ---AMPA_EXT---> E
		self.d_s_AMPA_EXT__CONN_0 = (-self.s_AMPA_EXT__CONN_0 / 0.002) 
		self.s_AMPA_EXT__CONN_0 = self.s_AMPA_EXT__CONN_0 + self.d_s_AMPA_EXT__CONN_0*self.dt
		self.s_AMPA_EXT__CONN_0 = self.s_AMPA_EXT__CONN_0 + self.gW[0][0]*self.stim.poisson_generator_CONN_0() 
		wj = np.multiply(self.gW[0][1], self.s_AMPA_EXT__CONN_0)
		self.I_syn[0:1600] = self.I_syn[0:1600] + np.einsum('i,ij->i', (self.V-self.VE)[0:1600], wj) 
	def integrate_CONN_1(self):
		# noiseI ---AMPA_EXT---> I
		self.d_s_AMPA_EXT__CONN_1 = (-self.s_AMPA_EXT__CONN_1 / 0.002) 
		self.s_AMPA_EXT__CONN_1 = self.s_AMPA_EXT__CONN_1 + self.d_s_AMPA_EXT__CONN_1*self.dt
		self.s_AMPA_EXT__CONN_1 = self.s_AMPA_EXT__CONN_1 + self.gW[1][0]*self.stim.poisson_generator_CONN_1() 
		wj = np.multiply(self.gW[1][1], self.s_AMPA_EXT__CONN_1)
		self.I_syn[1600:2000] = self.I_syn[1600:2000] + np.einsum('i,ij->i', (self.V-self.VE)[1600:2000], wj) 
	def integrate_CONN_2(self):
		# stimA ---AMPA_EXT_Subgroup_E---> [0:240]
		self.d_s_AMPA_EXT__CONN_2 = (-self.s_AMPA_EXT__CONN_2 / 0.002)
		self.s_AMPA_EXT__CONN_2 = self.s_AMPA_EXT__CONN_2 + self.d_s_AMPA_EXT__CONN_2*self.dt
		self.s_AMPA_EXT__CONN_2 = self.s_AMPA_EXT__CONN_2 + self.gW[2][0]*self.stim.poisson_generator_CONN_2() 
		wj = np.multiply(self.gW[2][1], self.s_AMPA_EXT__CONN_2)
		self.I_syn[0:1600][0:240] = self.I_syn[0:1600][0:240] + np.einsum('i,ij->i', (self.V-self.VE)[0:1600][0:240], wj) 
	def integrate_CONN_3(self):
		# stimB ---AMPA_EXT_Subgroup_E---> [240:480]
		self.d_s_AMPA_EXT__CONN_3 = (-self.s_AMPA_EXT__CONN_3 / 0.002)
		self.s_AMPA_EXT__CONN_3 = self.s_AMPA_EXT__CONN_3 + self.d_s_AMPA_EXT__CONN_3*self.dt
		self.s_AMPA_EXT__CONN_3 = self.s_AMPA_EXT__CONN_3 + self.gW[3][0]*self.stim.poisson_generator_CONN_3() 
		wj = np.multiply(self.gW[3][1], self.s_AMPA_EXT__CONN_3)
		self.I_syn[0:1600][240:480] = self.I_syn[0:1600][240:480] + np.einsum('i,ij->i', (self.V-self.VE)[0:1600][240:480], wj) 
	def integrate_CONN_4(self):
		# E ---AMPA---> E
		self.s_AMPA__CONN_4 = self.s_AMPA__CONN_4 + np.tile(self.last_spikes[0:1600], (1600,1))
		self.d_s_AMPA__CONN_4 = (-self.s_AMPA__CONN_4 / 0.002) 
		self.s_AMPA__CONN_4 = self.s_AMPA__CONN_4 + self.d_s_AMPA__CONN_4*self.dt
		wj = self.gW[4][0]*np.multiply(self.gW[4][1], self.s_AMPA__CONN_4)
		self.I_syn[0:1600] = self.I_syn[0:1600] + np.einsum('i,ij->i', (self.V-self.VE)[0:1600], wj) 
	def integrate_CONN_5(self):
		# E ---AMPA---> I
		self.s_AMPA__CONN_5 = self.s_AMPA__CONN_5 + np.tile(self.last_spikes[0:1600], (400,1))
		self.d_s_AMPA__CONN_5 = (-self.s_AMPA__CONN_5 / 0.002) 
		self.s_AMPA__CONN_5 = self.s_AMPA__CONN_5 + self.d_s_AMPA__CONN_5*self.dt
		wj = self.gW[5][0]*np.multiply(self.gW[5][1], self.s_AMPA__CONN_5)
		self.I_syn[1600:2000] = self.I_syn[1600:2000] + np.einsum('i,ij->i', (self.V-self.VE)[1600:2000], wj) 
	#%% firing control
	def integrate_and_fire(self, time_idx):
		is_in_rest = np.greater(self.t_ref, 0.0)
		self.t_ref = np.where(is_in_rest, self.t_ref - self.dt, self.t_ref)

		##% memb pot derivatives
		self.d_V = (-self.GL*(self.V - (-70e-3)) - self.I_syn) / self.CM
		integrated_V = self.V + self.d_V*self.dt
		self.V = np.where(np.logical_not(is_in_rest), integrated_V, self.V)
		is_fired = np.greater_equal(self.V, self.VT)
		self.V = np.where(is_fired, self.VR, self.V)
		self.t_ref = np.where(is_fired, self.TREF, self.t_ref)

		self.last_spikes = np.copy(is_fired)
		self.output_spikes[time_idx] = np.copy(is_fired)
	#%% forward func
	def forward(self, time_idx):
		self.I_syn = np.zeros(2000)
		self.integrate_CONN_0()
		self.integrate_CONN_1()
		self.integrate_CONN_2()
		self.integrate_CONN_3()
		self.integrate_CONN_4()
		self.integrate_CONN_5()
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