import numpy as np
from tqdm import tqdm
from elastica import *
from actuation_muscles import *
from tools import _aver, _aver_kernel


class SensoryFeedback:
	def __init__(self, env, callback_list: dict, step_skip: int, ramp_up_time=0.0):
		self.env = env
		self.callback_list = callback_list
		self.every = step_skip
		self.s = env.arm_param['s']
		self.ctrl_mag = np.zeros([3, env.arm_param['n_elem']+1])
		assert ramp_up_time >= 0.0
		if ramp_up_time == 0:
			self.ramp_up_time = 1e-14
		else:
			self.ramp_up_time = ramp_up_time
		self.count = 0

	def sensor(self, system, target):
		target_vector = target[:,None] - system.position_collection[:-1,:]
		self.dist = np.sqrt(np.einsum('in,in->n', target_vector, target_vector))
		self.min_idx = np.argmin(self.dist)
		self.s0 = self.s[self.min_idx]
		norm = _aver(self.dist)
		norm[norm==0] += 1e-16
		normalized_target_vector = _aver(target_vector) / norm
		tangent_vector = system.tangents[:-1,:]
		normal_vector = tangent_vector.copy()
		normal_vector[0, :] = -tangent_vector[1, :]
		normal_vector[1, :] = tangent_vector[0, :]
		self.sin_alpha = np.einsum('in,in->n', normalized_target_vector, normal_vector)

	def LM_choice(self, array):
		return np.where(array>=0), np.where(array<0)

	def TM_choice(self, array):
		return np.where(abs(array)<=1.)
	
	def feedback(self, time, system):
		error_feedback = self.sin_alpha #
		idx_top, idx_bottom = self.LM_choice(error_feedback)
		idx_central = self.TM_choice(error_feedback)
		sigma = 0.01 # 0.01
		if self.env.flags[3] == 'TM_reach':
			mag = 1. ## w/o transverse muscle
		else:
			mag = 0.
		steep = 300 # 500 # 200
		shift = 1.5 # 1.5 # 2.5
		## Ramp up the muscle torque
		factor = min(1.0, time / self.ramp_up_time)
		if self.s0 > self.s[10]:
			## longitudinal
			self.ctrl_mag[0,idx_top] = (-1 / (1 + np.exp(-steep * (self.s[idx_top] - (self.s0-shift*sigma)))) + 1) # shift*sigma*0
			self.ctrl_mag[1,idx_bottom] = (-1 / (1 + np.exp(-steep * (self.s[idx_bottom] - (self.s0-shift*sigma)))) + 1) # shift*sigma*0
			## transverse
			self.ctrl_mag[-1,idx_central] = (-1 / (1 + np.exp(-steep * (self.s[idx_central] - (self.s0-shift*sigma*0)))) + 1) * mag
		else:
			self.ctrl_mag[0,idx_top] = 1.
			self.ctrl_mag[1,idx_bottom] = 1.
		mag_feedback = _aver_kernel(abs(error_feedback))
		mag_feedback[0] = mag_feedback[1]
		mag_feedback[1] = 0.5 * (mag_feedback[0] + mag_feedback[2])
		self.ctrl_mag[:-1, :] *= factor * mag_feedback
		self.ctrl_mag = np.clip(self.ctrl_mag, 0, 1)
		self.ctrl_mag[-1, :] *= factor * (1 - mag_feedback**2)
	
	def sensory_feedback_law(self, time, system, target):
		self.sensor(system, target)
		self.feedback(time, system)
		self.callback()
		self.count += 1
		return self.ctrl_mag, self.s0
	
	def callback(self):
		if self.count % self.every == 0:
			# self.callback_list['u'].append(self.ctrl_mag.copy())
			self.callback_list['dist'].append(self.dist.copy())
			self.callback_list['angle'].append(self.sin_alpha.copy())
			self.callback_list['s_bar'].append(self.min_idx)