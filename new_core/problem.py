import numpy as np
from spayk.Stimuli import *
import pickle
def load_pickle(pkl_path):
	with open(pkl_path, 'rb') as handle:
		return pickle.load(handle)#%% Problem
class Problem:
	def __init__(self):
		self.runtime_path = '/home/gelenag/Dev/Spayk/new_core/first_run'
		self.neuron_dict = load_pickle('/home/gelenag/Dev/Spayk/new_core/first_run/neuron_dict.pickle')
		self.synapse_dict = load_pickle('/home/gelenag/Dev/Spayk/new_core/first_run/synapse_dict.pickle')

		self.dt = 0.1e-3

		self.spikes = []

		#%% Stimuli 
		self.stimuli = {'noiseE': PoissonSpikeTrain(5,5,(0, 1, 0.0001)),
'noiseI': PoissonSpikeTrain(5,5,(0, 1, 0.0001)),
'stimA': PoissonSpikeTrain(5,15,(0, 1, 0.0001)),
'stimB': PoissonSpikeTrain(5,30,(0, 1, 0.0001)),
		}
		self.current_stimuli = {'noiseE': self.stimuli['noiseE'].spikes[0],
'noiseI': self.stimuli['noiseI'].spikes[0],
'stimA': self.stimuli['stimA'].spikes[0],
'stimB': self.stimuli['stimB'].spikes[0],
'E': np.zeros(16),
'I': np.zeros(4),
		}
		#%% neuron memb. potens.
		# >>>>>>>>> lif variables
		self.V = np.zeros(20)
		self.t_ref = np.zeros(20)
		self.I_syn = np.zeros(20)
		self.neuron_GLs = self.neuron_dict['GLs']
		self.neuron_VLs = self.neuron_dict['VLs']
		self.neuron_CMs = self.neuron_dict['CMs']
		# >>>>>>>>> for neuron E0
		self.s_AMPA_E0 = np.zeros(16)
		self.s_NMDA_E0 = np.zeros(16)
		self.x_NMDA_E0 = np.zeros(16)
		self.s_GABA_E0 = np.zeros(4)
		self.s_AMPA_EXT_E0 = np.zeros(10)
		# >>>>>>>>> for neuron E1
		self.s_AMPA_E1 = np.zeros(16)
		self.s_NMDA_E1 = np.zeros(16)
		self.x_NMDA_E1 = np.zeros(16)
		self.s_GABA_E1 = np.zeros(4)
		self.s_AMPA_EXT_E1 = np.zeros(10)
		# >>>>>>>>> for neuron E2
		self.s_AMPA_E2 = np.zeros(16)
		self.s_NMDA_E2 = np.zeros(16)
		self.x_NMDA_E2 = np.zeros(16)
		self.s_GABA_E2 = np.zeros(4)
		self.s_AMPA_EXT_E2 = np.zeros(10)
		# >>>>>>>>> for neuron E3
		self.s_AMPA_E3 = np.zeros(16)
		self.s_NMDA_E3 = np.zeros(16)
		self.x_NMDA_E3 = np.zeros(16)
		self.s_GABA_E3 = np.zeros(4)
		self.s_AMPA_EXT_E3 = np.zeros(10)
		# >>>>>>>>> for neuron E4
		self.s_AMPA_E4 = np.zeros(16)
		self.s_NMDA_E4 = np.zeros(16)
		self.x_NMDA_E4 = np.zeros(16)
		self.s_GABA_E4 = np.zeros(4)
		self.s_AMPA_EXT_E4 = np.zeros(10)
		# >>>>>>>>> for neuron E5
		self.s_AMPA_E5 = np.zeros(16)
		self.s_NMDA_E5 = np.zeros(16)
		self.x_NMDA_E5 = np.zeros(16)
		self.s_GABA_E5 = np.zeros(4)
		self.s_AMPA_EXT_E5 = np.zeros(10)
		# >>>>>>>>> for neuron E6
		self.s_AMPA_E6 = np.zeros(16)
		self.s_NMDA_E6 = np.zeros(16)
		self.x_NMDA_E6 = np.zeros(16)
		self.s_GABA_E6 = np.zeros(4)
		self.s_AMPA_EXT_E6 = np.zeros(10)
		# >>>>>>>>> for neuron E7
		self.s_AMPA_E7 = np.zeros(16)
		self.s_NMDA_E7 = np.zeros(16)
		self.x_NMDA_E7 = np.zeros(16)
		self.s_GABA_E7 = np.zeros(4)
		self.s_AMPA_EXT_E7 = np.zeros(10)
		# >>>>>>>>> for neuron E8
		self.s_AMPA_E8 = np.zeros(16)
		self.s_NMDA_E8 = np.zeros(16)
		self.x_NMDA_E8 = np.zeros(16)
		self.s_GABA_E8 = np.zeros(4)
		self.s_AMPA_EXT_E8 = np.zeros(5)
		# >>>>>>>>> for neuron E9
		self.s_AMPA_E9 = np.zeros(16)
		self.s_NMDA_E9 = np.zeros(16)
		self.x_NMDA_E9 = np.zeros(16)
		self.s_GABA_E9 = np.zeros(4)
		self.s_AMPA_EXT_E9 = np.zeros(5)
		# >>>>>>>>> for neuron E10
		self.s_AMPA_E10 = np.zeros(16)
		self.s_NMDA_E10 = np.zeros(16)
		self.x_NMDA_E10 = np.zeros(16)
		self.s_GABA_E10 = np.zeros(4)
		self.s_AMPA_EXT_E10 = np.zeros(5)
		# >>>>>>>>> for neuron E11
		self.s_AMPA_E11 = np.zeros(16)
		self.s_NMDA_E11 = np.zeros(16)
		self.x_NMDA_E11 = np.zeros(16)
		self.s_GABA_E11 = np.zeros(4)
		self.s_AMPA_EXT_E11 = np.zeros(5)
		# >>>>>>>>> for neuron E12
		self.s_AMPA_E12 = np.zeros(16)
		self.s_NMDA_E12 = np.zeros(16)
		self.x_NMDA_E12 = np.zeros(16)
		self.s_GABA_E12 = np.zeros(4)
		self.s_AMPA_EXT_E12 = np.zeros(5)
		# >>>>>>>>> for neuron E13
		self.s_AMPA_E13 = np.zeros(16)
		self.s_NMDA_E13 = np.zeros(16)
		self.x_NMDA_E13 = np.zeros(16)
		self.s_GABA_E13 = np.zeros(4)
		self.s_AMPA_EXT_E13 = np.zeros(5)
		# >>>>>>>>> for neuron E14
		self.s_AMPA_E14 = np.zeros(16)
		self.s_NMDA_E14 = np.zeros(16)
		self.x_NMDA_E14 = np.zeros(16)
		self.s_GABA_E14 = np.zeros(4)
		self.s_AMPA_EXT_E14 = np.zeros(5)
		# >>>>>>>>> for neuron E15
		self.s_AMPA_E15 = np.zeros(16)
		self.s_NMDA_E15 = np.zeros(16)
		self.x_NMDA_E15 = np.zeros(16)
		self.s_GABA_E15 = np.zeros(4)
		self.s_AMPA_EXT_E15 = np.zeros(5)
		# >>>>>>>>> for neuron I0
		self.s_AMPA_I0 = np.zeros(16)
		self.s_NMDA_I0 = np.zeros(16)
		self.x_NMDA_I0 = np.zeros(16)
		self.s_GABA_I0 = np.zeros(4)
		self.s_AMPA_EXT_I0 = np.zeros(5)
		# >>>>>>>>> for neuron I1
		self.s_AMPA_I1 = np.zeros(16)
		self.s_NMDA_I1 = np.zeros(16)
		self.x_NMDA_I1 = np.zeros(16)
		self.s_GABA_I1 = np.zeros(4)
		self.s_AMPA_EXT_I1 = np.zeros(5)
		# >>>>>>>>> for neuron I2
		self.s_AMPA_I2 = np.zeros(16)
		self.s_NMDA_I2 = np.zeros(16)
		self.x_NMDA_I2 = np.zeros(16)
		self.s_GABA_I2 = np.zeros(4)
		self.s_AMPA_EXT_I2 = np.zeros(5)
		# >>>>>>>>> for neuron I3
		self.s_AMPA_I3 = np.zeros(16)
		self.s_NMDA_I3 = np.zeros(16)
		self.x_NMDA_I3 = np.zeros(16)
		self.s_GABA_I3 = np.zeros(4)
		self.s_AMPA_EXT_I3 = np.zeros(5)
		# derivative reset func
		self.dx_reset()
	#%% derivative reset func
	def dx_reset(self):
		self.d_V = np.zeros(20)
		# >>>>>>>>> for neuron E0
		self.d_s_AMPA_E0 = np.zeros(16)
		self.d_s_NMDA_E0 = np.zeros(16)
		self.d_x_NMDA_E0 = np.zeros(16)
		self.d_s_GABA_E0 = np.zeros(4)
		self.d_s_AMPA_EXT_E0 = np.zeros(10)
		# >>>>>>>>> for neuron E1
		self.d_s_AMPA_E1 = np.zeros(16)
		self.d_s_NMDA_E1 = np.zeros(16)
		self.d_x_NMDA_E1 = np.zeros(16)
		self.d_s_GABA_E1 = np.zeros(4)
		self.d_s_AMPA_EXT_E1 = np.zeros(10)
		# >>>>>>>>> for neuron E2
		self.d_s_AMPA_E2 = np.zeros(16)
		self.d_s_NMDA_E2 = np.zeros(16)
		self.d_x_NMDA_E2 = np.zeros(16)
		self.d_s_GABA_E2 = np.zeros(4)
		self.d_s_AMPA_EXT_E2 = np.zeros(10)
		# >>>>>>>>> for neuron E3
		self.d_s_AMPA_E3 = np.zeros(16)
		self.d_s_NMDA_E3 = np.zeros(16)
		self.d_x_NMDA_E3 = np.zeros(16)
		self.d_s_GABA_E3 = np.zeros(4)
		self.d_s_AMPA_EXT_E3 = np.zeros(10)
		# >>>>>>>>> for neuron E4
		self.d_s_AMPA_E4 = np.zeros(16)
		self.d_s_NMDA_E4 = np.zeros(16)
		self.d_x_NMDA_E4 = np.zeros(16)
		self.d_s_GABA_E4 = np.zeros(4)
		self.d_s_AMPA_EXT_E4 = np.zeros(10)
		# >>>>>>>>> for neuron E5
		self.d_s_AMPA_E5 = np.zeros(16)
		self.d_s_NMDA_E5 = np.zeros(16)
		self.d_x_NMDA_E5 = np.zeros(16)
		self.d_s_GABA_E5 = np.zeros(4)
		self.d_s_AMPA_EXT_E5 = np.zeros(10)
		# >>>>>>>>> for neuron E6
		self.d_s_AMPA_E6 = np.zeros(16)
		self.d_s_NMDA_E6 = np.zeros(16)
		self.d_x_NMDA_E6 = np.zeros(16)
		self.d_s_GABA_E6 = np.zeros(4)
		self.d_s_AMPA_EXT_E6 = np.zeros(10)
		# >>>>>>>>> for neuron E7
		self.d_s_AMPA_E7 = np.zeros(16)
		self.d_s_NMDA_E7 = np.zeros(16)
		self.d_x_NMDA_E7 = np.zeros(16)
		self.d_s_GABA_E7 = np.zeros(4)
		self.d_s_AMPA_EXT_E7 = np.zeros(10)
		# >>>>>>>>> for neuron E8
		self.d_s_AMPA_E8 = np.zeros(16)
		self.d_s_NMDA_E8 = np.zeros(16)
		self.d_x_NMDA_E8 = np.zeros(16)
		self.d_s_GABA_E8 = np.zeros(4)
		self.d_s_AMPA_EXT_E8 = np.zeros(5)
		# >>>>>>>>> for neuron E9
		self.d_s_AMPA_E9 = np.zeros(16)
		self.d_s_NMDA_E9 = np.zeros(16)
		self.d_x_NMDA_E9 = np.zeros(16)
		self.d_s_GABA_E9 = np.zeros(4)
		self.d_s_AMPA_EXT_E9 = np.zeros(5)
		# >>>>>>>>> for neuron E10
		self.d_s_AMPA_E10 = np.zeros(16)
		self.d_s_NMDA_E10 = np.zeros(16)
		self.d_x_NMDA_E10 = np.zeros(16)
		self.d_s_GABA_E10 = np.zeros(4)
		self.d_s_AMPA_EXT_E10 = np.zeros(5)
		# >>>>>>>>> for neuron E11
		self.d_s_AMPA_E11 = np.zeros(16)
		self.d_s_NMDA_E11 = np.zeros(16)
		self.d_x_NMDA_E11 = np.zeros(16)
		self.d_s_GABA_E11 = np.zeros(4)
		self.d_s_AMPA_EXT_E11 = np.zeros(5)
		# >>>>>>>>> for neuron E12
		self.d_s_AMPA_E12 = np.zeros(16)
		self.d_s_NMDA_E12 = np.zeros(16)
		self.d_x_NMDA_E12 = np.zeros(16)
		self.d_s_GABA_E12 = np.zeros(4)
		self.d_s_AMPA_EXT_E12 = np.zeros(5)
		# >>>>>>>>> for neuron E13
		self.d_s_AMPA_E13 = np.zeros(16)
		self.d_s_NMDA_E13 = np.zeros(16)
		self.d_x_NMDA_E13 = np.zeros(16)
		self.d_s_GABA_E13 = np.zeros(4)
		self.d_s_AMPA_EXT_E13 = np.zeros(5)
		# >>>>>>>>> for neuron E14
		self.d_s_AMPA_E14 = np.zeros(16)
		self.d_s_NMDA_E14 = np.zeros(16)
		self.d_x_NMDA_E14 = np.zeros(16)
		self.d_s_GABA_E14 = np.zeros(4)
		self.d_s_AMPA_EXT_E14 = np.zeros(5)
		# >>>>>>>>> for neuron E15
		self.d_s_AMPA_E15 = np.zeros(16)
		self.d_s_NMDA_E15 = np.zeros(16)
		self.d_x_NMDA_E15 = np.zeros(16)
		self.d_s_GABA_E15 = np.zeros(4)
		self.d_s_AMPA_EXT_E15 = np.zeros(5)
		# >>>>>>>>> for neuron I0
		self.d_s_AMPA_I0 = np.zeros(16)
		self.d_s_NMDA_I0 = np.zeros(16)
		self.d_x_NMDA_I0 = np.zeros(16)
		self.d_s_GABA_I0 = np.zeros(4)
		self.d_s_AMPA_EXT_I0 = np.zeros(5)
		# >>>>>>>>> for neuron I1
		self.d_s_AMPA_I1 = np.zeros(16)
		self.d_s_NMDA_I1 = np.zeros(16)
		self.d_x_NMDA_I1 = np.zeros(16)
		self.d_s_GABA_I1 = np.zeros(4)
		self.d_s_AMPA_EXT_I1 = np.zeros(5)
		# >>>>>>>>> for neuron I2
		self.d_s_AMPA_I2 = np.zeros(16)
		self.d_s_NMDA_I2 = np.zeros(16)
		self.d_x_NMDA_I2 = np.zeros(16)
		self.d_s_GABA_I2 = np.zeros(4)
		self.d_s_AMPA_EXT_I2 = np.zeros(5)
		# >>>>>>>>> for neuron I3
		self.d_s_AMPA_I3 = np.zeros(16)
		self.d_s_NMDA_I3 = np.zeros(16)
		self.d_x_NMDA_I3 = np.zeros(16)
		self.d_s_GABA_I3 = np.zeros(4)
		self.d_s_AMPA_EXT_I3 = np.zeros(5)
	#%% euler integration func
	def integrate_all_euler(self):
		self.V += self.d_V*self.dt
		# >>>>>>>>> for neuron E0
		self.s_AMPA_E0 += self.d_s_AMPA_E0*self.dt
		self.s_NMDA_E0 += self.d_s_NMDA_E0*self.dt
		self.x_NMDA_E0 += self.d_x_NMDA_E0*self.dt
		self.s_GABA_E0 += self.d_s_GABA_E0*self.dt
		self.s_AMPA_EXT_E0 += self.d_s_AMPA_EXT_E0*self.dt
		# >>>>>>>>> for neuron E1
		self.s_AMPA_E1 += self.d_s_AMPA_E1*self.dt
		self.s_NMDA_E1 += self.d_s_NMDA_E1*self.dt
		self.x_NMDA_E1 += self.d_x_NMDA_E1*self.dt
		self.s_GABA_E1 += self.d_s_GABA_E1*self.dt
		self.s_AMPA_EXT_E1 += self.d_s_AMPA_EXT_E1*self.dt
		# >>>>>>>>> for neuron E2
		self.s_AMPA_E2 += self.d_s_AMPA_E2*self.dt
		self.s_NMDA_E2 += self.d_s_NMDA_E2*self.dt
		self.x_NMDA_E2 += self.d_x_NMDA_E2*self.dt
		self.s_GABA_E2 += self.d_s_GABA_E2*self.dt
		self.s_AMPA_EXT_E2 += self.d_s_AMPA_EXT_E2*self.dt
		# >>>>>>>>> for neuron E3
		self.s_AMPA_E3 += self.d_s_AMPA_E3*self.dt
		self.s_NMDA_E3 += self.d_s_NMDA_E3*self.dt
		self.x_NMDA_E3 += self.d_x_NMDA_E3*self.dt
		self.s_GABA_E3 += self.d_s_GABA_E3*self.dt
		self.s_AMPA_EXT_E3 += self.d_s_AMPA_EXT_E3*self.dt
		# >>>>>>>>> for neuron E4
		self.s_AMPA_E4 += self.d_s_AMPA_E4*self.dt
		self.s_NMDA_E4 += self.d_s_NMDA_E4*self.dt
		self.x_NMDA_E4 += self.d_x_NMDA_E4*self.dt
		self.s_GABA_E4 += self.d_s_GABA_E4*self.dt
		self.s_AMPA_EXT_E4 += self.d_s_AMPA_EXT_E4*self.dt
		# >>>>>>>>> for neuron E5
		self.s_AMPA_E5 += self.d_s_AMPA_E5*self.dt
		self.s_NMDA_E5 += self.d_s_NMDA_E5*self.dt
		self.x_NMDA_E5 += self.d_x_NMDA_E5*self.dt
		self.s_GABA_E5 += self.d_s_GABA_E5*self.dt
		self.s_AMPA_EXT_E5 += self.d_s_AMPA_EXT_E5*self.dt
		# >>>>>>>>> for neuron E6
		self.s_AMPA_E6 += self.d_s_AMPA_E6*self.dt
		self.s_NMDA_E6 += self.d_s_NMDA_E6*self.dt
		self.x_NMDA_E6 += self.d_x_NMDA_E6*self.dt
		self.s_GABA_E6 += self.d_s_GABA_E6*self.dt
		self.s_AMPA_EXT_E6 += self.d_s_AMPA_EXT_E6*self.dt
		# >>>>>>>>> for neuron E7
		self.s_AMPA_E7 += self.d_s_AMPA_E7*self.dt
		self.s_NMDA_E7 += self.d_s_NMDA_E7*self.dt
		self.x_NMDA_E7 += self.d_x_NMDA_E7*self.dt
		self.s_GABA_E7 += self.d_s_GABA_E7*self.dt
		self.s_AMPA_EXT_E7 += self.d_s_AMPA_EXT_E7*self.dt
		# >>>>>>>>> for neuron E8
		self.s_AMPA_E8 += self.d_s_AMPA_E8*self.dt
		self.s_NMDA_E8 += self.d_s_NMDA_E8*self.dt
		self.x_NMDA_E8 += self.d_x_NMDA_E8*self.dt
		self.s_GABA_E8 += self.d_s_GABA_E8*self.dt
		self.s_AMPA_EXT_E8 += self.d_s_AMPA_EXT_E8*self.dt
		# >>>>>>>>> for neuron E9
		self.s_AMPA_E9 += self.d_s_AMPA_E9*self.dt
		self.s_NMDA_E9 += self.d_s_NMDA_E9*self.dt
		self.x_NMDA_E9 += self.d_x_NMDA_E9*self.dt
		self.s_GABA_E9 += self.d_s_GABA_E9*self.dt
		self.s_AMPA_EXT_E9 += self.d_s_AMPA_EXT_E9*self.dt
		# >>>>>>>>> for neuron E10
		self.s_AMPA_E10 += self.d_s_AMPA_E10*self.dt
		self.s_NMDA_E10 += self.d_s_NMDA_E10*self.dt
		self.x_NMDA_E10 += self.d_x_NMDA_E10*self.dt
		self.s_GABA_E10 += self.d_s_GABA_E10*self.dt
		self.s_AMPA_EXT_E10 += self.d_s_AMPA_EXT_E10*self.dt
		# >>>>>>>>> for neuron E11
		self.s_AMPA_E11 += self.d_s_AMPA_E11*self.dt
		self.s_NMDA_E11 += self.d_s_NMDA_E11*self.dt
		self.x_NMDA_E11 += self.d_x_NMDA_E11*self.dt
		self.s_GABA_E11 += self.d_s_GABA_E11*self.dt
		self.s_AMPA_EXT_E11 += self.d_s_AMPA_EXT_E11*self.dt
		# >>>>>>>>> for neuron E12
		self.s_AMPA_E12 += self.d_s_AMPA_E12*self.dt
		self.s_NMDA_E12 += self.d_s_NMDA_E12*self.dt
		self.x_NMDA_E12 += self.d_x_NMDA_E12*self.dt
		self.s_GABA_E12 += self.d_s_GABA_E12*self.dt
		self.s_AMPA_EXT_E12 += self.d_s_AMPA_EXT_E12*self.dt
		# >>>>>>>>> for neuron E13
		self.s_AMPA_E13 += self.d_s_AMPA_E13*self.dt
		self.s_NMDA_E13 += self.d_s_NMDA_E13*self.dt
		self.x_NMDA_E13 += self.d_x_NMDA_E13*self.dt
		self.s_GABA_E13 += self.d_s_GABA_E13*self.dt
		self.s_AMPA_EXT_E13 += self.d_s_AMPA_EXT_E13*self.dt
		# >>>>>>>>> for neuron E14
		self.s_AMPA_E14 += self.d_s_AMPA_E14*self.dt
		self.s_NMDA_E14 += self.d_s_NMDA_E14*self.dt
		self.x_NMDA_E14 += self.d_x_NMDA_E14*self.dt
		self.s_GABA_E14 += self.d_s_GABA_E14*self.dt
		self.s_AMPA_EXT_E14 += self.d_s_AMPA_EXT_E14*self.dt
		# >>>>>>>>> for neuron E15
		self.s_AMPA_E15 += self.d_s_AMPA_E15*self.dt
		self.s_NMDA_E15 += self.d_s_NMDA_E15*self.dt
		self.x_NMDA_E15 += self.d_x_NMDA_E15*self.dt
		self.s_GABA_E15 += self.d_s_GABA_E15*self.dt
		self.s_AMPA_EXT_E15 += self.d_s_AMPA_EXT_E15*self.dt
		# >>>>>>>>> for neuron I0
		self.s_AMPA_I0 += self.d_s_AMPA_I0*self.dt
		self.s_NMDA_I0 += self.d_s_NMDA_I0*self.dt
		self.x_NMDA_I0 += self.d_x_NMDA_I0*self.dt
		self.s_GABA_I0 += self.d_s_GABA_I0*self.dt
		self.s_AMPA_EXT_I0 += self.d_s_AMPA_EXT_I0*self.dt
		# >>>>>>>>> for neuron I1
		self.s_AMPA_I1 += self.d_s_AMPA_I1*self.dt
		self.s_NMDA_I1 += self.d_s_NMDA_I1*self.dt
		self.x_NMDA_I1 += self.d_x_NMDA_I1*self.dt
		self.s_GABA_I1 += self.d_s_GABA_I1*self.dt
		self.s_AMPA_EXT_I1 += self.d_s_AMPA_EXT_I1*self.dt
		# >>>>>>>>> for neuron I2
		self.s_AMPA_I2 += self.d_s_AMPA_I2*self.dt
		self.s_NMDA_I2 += self.d_s_NMDA_I2*self.dt
		self.x_NMDA_I2 += self.d_x_NMDA_I2*self.dt
		self.s_GABA_I2 += self.d_s_GABA_I2*self.dt
		self.s_AMPA_EXT_I2 += self.d_s_AMPA_EXT_I2*self.dt
		# >>>>>>>>> for neuron I3
		self.s_AMPA_I3 += self.d_s_AMPA_I3*self.dt
		self.s_NMDA_I3 += self.d_s_NMDA_I3*self.dt
		self.x_NMDA_I3 += self.d_x_NMDA_I3*self.dt
		self.s_GABA_I3 += self.d_s_GABA_I3*self.dt
		self.s_AMPA_EXT_I3 += self.d_s_AMPA_EXT_I3*self.dt
	#%% derivative func
	def calculate_dxdt_all(self):
		self.dx_reset()
		##% memb pot derivatives
		self.d_V = (-self.neuron_GLs*(self.V - self.neuron_VLs) - self.I_syn) / self.neuron_CMs
		##% channel derivatives
		# E ---AMPA---> E
		self.d_s_AMPA_E0 += (-self.s_AMPA_E0 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E1 += (-self.s_AMPA_E1 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E2 += (-self.s_AMPA_E2 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E3 += (-self.s_AMPA_E3 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E4 += (-self.s_AMPA_E4 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E5 += (-self.s_AMPA_E5 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E6 += (-self.s_AMPA_E6 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E7 += (-self.s_AMPA_E7 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E8 += (-self.s_AMPA_E8 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E9 += (-self.s_AMPA_E9 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E10 += (-self.s_AMPA_E10 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E11 += (-self.s_AMPA_E11 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E12 += (-self.s_AMPA_E12 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E13 += (-self.s_AMPA_E13 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E14 += (-self.s_AMPA_E14 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_E15 += (-self.s_AMPA_E15 / 0.002) + self.current_stimuli['E'] 
		# E ---NMDA---> E
		self.d_x_NMDA_E0 += (-self.x_NMDA_E0 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E0 += (-self.s_NMDA_E0 / 0.1) + 500.0*self.x_NMDA_E0*(1 - self.s_NMDA_E0) 
		self.d_x_NMDA_E1 += (-self.x_NMDA_E1 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E1 += (-self.s_NMDA_E1 / 0.1) + 500.0*self.x_NMDA_E1*(1 - self.s_NMDA_E1) 
		self.d_x_NMDA_E2 += (-self.x_NMDA_E2 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E2 += (-self.s_NMDA_E2 / 0.1) + 500.0*self.x_NMDA_E2*(1 - self.s_NMDA_E2) 
		self.d_x_NMDA_E3 += (-self.x_NMDA_E3 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E3 += (-self.s_NMDA_E3 / 0.1) + 500.0*self.x_NMDA_E3*(1 - self.s_NMDA_E3) 
		self.d_x_NMDA_E4 += (-self.x_NMDA_E4 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E4 += (-self.s_NMDA_E4 / 0.1) + 500.0*self.x_NMDA_E4*(1 - self.s_NMDA_E4) 
		self.d_x_NMDA_E5 += (-self.x_NMDA_E5 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E5 += (-self.s_NMDA_E5 / 0.1) + 500.0*self.x_NMDA_E5*(1 - self.s_NMDA_E5) 
		self.d_x_NMDA_E6 += (-self.x_NMDA_E6 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E6 += (-self.s_NMDA_E6 / 0.1) + 500.0*self.x_NMDA_E6*(1 - self.s_NMDA_E6) 
		self.d_x_NMDA_E7 += (-self.x_NMDA_E7 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E7 += (-self.s_NMDA_E7 / 0.1) + 500.0*self.x_NMDA_E7*(1 - self.s_NMDA_E7) 
		self.d_x_NMDA_E8 += (-self.x_NMDA_E8 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E8 += (-self.s_NMDA_E8 / 0.1) + 500.0*self.x_NMDA_E8*(1 - self.s_NMDA_E8) 
		self.d_x_NMDA_E9 += (-self.x_NMDA_E9 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E9 += (-self.s_NMDA_E9 / 0.1) + 500.0*self.x_NMDA_E9*(1 - self.s_NMDA_E9) 
		self.d_x_NMDA_E10 += (-self.x_NMDA_E10 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E10 += (-self.s_NMDA_E10 / 0.1) + 500.0*self.x_NMDA_E10*(1 - self.s_NMDA_E10) 
		self.d_x_NMDA_E11 += (-self.x_NMDA_E11 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E11 += (-self.s_NMDA_E11 / 0.1) + 500.0*self.x_NMDA_E11*(1 - self.s_NMDA_E11) 
		self.d_x_NMDA_E12 += (-self.x_NMDA_E12 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E12 += (-self.s_NMDA_E12 / 0.1) + 500.0*self.x_NMDA_E12*(1 - self.s_NMDA_E12) 
		self.d_x_NMDA_E13 += (-self.x_NMDA_E13 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E13 += (-self.s_NMDA_E13 / 0.1) + 500.0*self.x_NMDA_E13*(1 - self.s_NMDA_E13) 
		self.d_x_NMDA_E14 += (-self.x_NMDA_E14 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E14 += (-self.s_NMDA_E14 / 0.1) + 500.0*self.x_NMDA_E14*(1 - self.s_NMDA_E14) 
		self.d_x_NMDA_E15 += (-self.x_NMDA_E15 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_E15 += (-self.s_NMDA_E15 / 0.1) + 500.0*self.x_NMDA_E15*(1 - self.s_NMDA_E15) 
		# E ---AMPA---> I
		self.d_s_AMPA_I0 += (-self.s_AMPA_I0 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_I1 += (-self.s_AMPA_I1 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_I2 += (-self.s_AMPA_I2 / 0.002) + self.current_stimuli['E'] 
		self.d_s_AMPA_I3 += (-self.s_AMPA_I3 / 0.002) + self.current_stimuli['E'] 
		# E ---NMDA---> I
		self.d_x_NMDA_I0 += (-self.x_NMDA_I0 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_I0 += (-self.s_NMDA_I0 / 0.1) + 500.0*self.x_NMDA_I0*(1 - self.s_NMDA_I0) 
		self.d_x_NMDA_I1 += (-self.x_NMDA_I1 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_I1 += (-self.s_NMDA_I1 / 0.1) + 500.0*self.x_NMDA_I1*(1 - self.s_NMDA_I1) 
		self.d_x_NMDA_I2 += (-self.x_NMDA_I2 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_I2 += (-self.s_NMDA_I2 / 0.1) + 500.0*self.x_NMDA_I2*(1 - self.s_NMDA_I2) 
		self.d_x_NMDA_I3 += (-self.x_NMDA_I3 / 0.002) + self.current_stimuli['E'] 
		self.d_s_NMDA_I3 += (-self.s_NMDA_I3 / 0.1) + 500.0*self.x_NMDA_I3*(1 - self.s_NMDA_I3) 
		# I ---GABA---> E
		self.d_s_GABA_E0 += (-self.s_GABA_E0 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E1 += (-self.s_GABA_E1 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E2 += (-self.s_GABA_E2 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E3 += (-self.s_GABA_E3 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E4 += (-self.s_GABA_E4 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E5 += (-self.s_GABA_E5 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E6 += (-self.s_GABA_E6 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E7 += (-self.s_GABA_E7 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E8 += (-self.s_GABA_E8 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E9 += (-self.s_GABA_E9 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E10 += (-self.s_GABA_E10 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E11 += (-self.s_GABA_E11 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E12 += (-self.s_GABA_E12 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E13 += (-self.s_GABA_E13 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E14 += (-self.s_GABA_E14 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_E15 += (-self.s_GABA_E15 / 0.005) + self.current_stimuli['I'] 
		# I ---GABA---> I
		self.d_s_GABA_I0 += (-self.s_GABA_I0 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_I1 += (-self.s_GABA_I1 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_I2 += (-self.s_GABA_I2 / 0.005) + self.current_stimuli['I'] 
		self.d_s_GABA_I3 += (-self.s_GABA_I3 / 0.005) + self.current_stimuli['I'] 
		# noiseE ---AMPA_EXT---> E
		self.d_s_AMPA_EXT_E0[0:5] += (-self.s_AMPA_EXT_E0[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E1[0:5] += (-self.s_AMPA_EXT_E1[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E2[0:5] += (-self.s_AMPA_EXT_E2[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E3[0:5] += (-self.s_AMPA_EXT_E3[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E4[0:5] += (-self.s_AMPA_EXT_E4[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E5[0:5] += (-self.s_AMPA_EXT_E5[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E6[0:5] += (-self.s_AMPA_EXT_E6[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E7[0:5] += (-self.s_AMPA_EXT_E7[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E8[0:5] += (-self.s_AMPA_EXT_E8[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E9[0:5] += (-self.s_AMPA_EXT_E9[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E10[0:5] += (-self.s_AMPA_EXT_E10[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E11[0:5] += (-self.s_AMPA_EXT_E11[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E12[0:5] += (-self.s_AMPA_EXT_E12[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E13[0:5] += (-self.s_AMPA_EXT_E13[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E14[0:5] += (-self.s_AMPA_EXT_E14[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		self.d_s_AMPA_EXT_E15[0:5] += (-self.s_AMPA_EXT_E15[0:5] / 0.002) + self.current_stimuli['noiseE'][0:5]  
		# noiseI ---AMPA_EXT---> I
		self.d_s_AMPA_EXT_I0[0:5] += (-self.s_AMPA_EXT_I0[0:5] / 0.002) + self.current_stimuli['noiseI'][0:5]  
		self.d_s_AMPA_EXT_I1[0:5] += (-self.s_AMPA_EXT_I1[0:5] / 0.002) + self.current_stimuli['noiseI'][0:5]  
		self.d_s_AMPA_EXT_I2[0:5] += (-self.s_AMPA_EXT_I2[0:5] / 0.002) + self.current_stimuli['noiseI'][0:5]  
		self.d_s_AMPA_EXT_I3[0:5] += (-self.s_AMPA_EXT_I3[0:5] / 0.002) + self.current_stimuli['noiseI'][0:5]  
		# stimA ---AMPA_EXT---subgroup of---> E
		self.d_s_AMPA_EXT_E0[5:10] += (-self.s_AMPA_EXT_E0[5:10] / 0.002) + self.current_stimuli['stimA'][5:10]  
		self.d_s_AMPA_EXT_E1[5:10] += (-self.s_AMPA_EXT_E1[5:10] / 0.002) + self.current_stimuli['stimA'][5:10]  
		self.d_s_AMPA_EXT_E2[5:10] += (-self.s_AMPA_EXT_E2[5:10] / 0.002) + self.current_stimuli['stimA'][5:10]  
		self.d_s_AMPA_EXT_E3[5:10] += (-self.s_AMPA_EXT_E3[5:10] / 0.002) + self.current_stimuli['stimA'][5:10]  
		# stimB ---AMPA_EXT---subgroup of---> E
		self.d_s_AMPA_EXT_E4[5:10] += (-self.s_AMPA_EXT_E4[5:10] / 0.002) + self.current_stimuli['stimB'][5:10]  
		self.d_s_AMPA_EXT_E5[5:10] += (-self.s_AMPA_EXT_E5[5:10] / 0.002) + self.current_stimuli['stimB'][5:10]  
		self.d_s_AMPA_EXT_E6[5:10] += (-self.s_AMPA_EXT_E6[5:10] / 0.002) + self.current_stimuli['stimB'][5:10]  
		self.d_s_AMPA_EXT_E7[5:10] += (-self.s_AMPA_EXT_E7[5:10] / 0.002) + self.current_stimuli['stimB'][5:10]  
	#%% synaptic current calculation func
	def calculate_synaptic_currents(self):
		# E ---AMPA---> E
		self.I_Syn[0] += 2.1e-09*(self.V[0]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[1] += 2.1e-09*(self.V[1]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[2] += 2.1e-09*(self.V[2]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[3] += 2.1e-09*(self.V[3]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[4] += 2.1e-09*(self.V[4]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[5] += 2.1e-09*(self.V[5]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[6] += 2.1e-09*(self.V[6]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[7] += 2.1e-09*(self.V[7]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[8] += 2.1e-09*(self.V[8]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[9] += 2.1e-09*(self.V[9]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[10] += 2.1e-09*(self.V[10]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[11] += 2.1e-09*(self.V[11]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[12] += 2.1e-09*(self.V[12]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[13] += 2.1e-09*(self.V[13]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[14] += 2.1e-09*(self.V[14]-0)*(self.synapse_dict['connection_list'][0]) 
		self.I_Syn[15] += 2.1e-09*(self.V[15]-0)*(self.synapse_dict['connection_list'][0]) 
		# E ---AMPA---> I
		self.I_Syn[16] += 4e-09*(self.V[16]-0)*(self.synapse_dict['connection_list'][2]) 
		self.I_Syn[17] += 4e-09*(self.V[17]-0)*(self.synapse_dict['connection_list'][2]) 
		self.I_Syn[18] += 4e-09*(self.V[18]-0)*(self.synapse_dict['connection_list'][2]) 
		self.I_Syn[19] += 4e-09*(self.V[19]-0)*(self.synapse_dict['connection_list'][2]) 
