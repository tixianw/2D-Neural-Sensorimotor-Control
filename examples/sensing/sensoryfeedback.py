"""
Created on Feb Sep 26, 2023
@author: tixianw2
"""
import numpy as np
from tqdm import tqdm
from elastica import *
from actuation_muscles import *
from tools import _aver, cum_integral, softargmax
from scipy.integrate import odeint

def f(y,t,k,dt):
	dist = y[0]
	alpha = y[1]
	idx = int(t / dt)
	y0s = -np.cos(alpha)
	if idx>=5:
		idx = 4
	y1s = -k[idx] + 1/dist * np.sin(alpha)
	return np.array([y0s, y1s])

def f1(y,k):
	dist = y[0]
	alpha = y[1]
	y0s = -np.cos(alpha)
	y1s = -k + 1/dist * np.sin(alpha)
	return np.array([y0s, y1s])

def solve(f, y0, ss, kk, ds):
	alpha = [y0[1]]
	y = y0.copy()
	for i in range(len(ss)-1):
		y += f1(y, kk[i]) * ds
		alpha.append(y[1])
	return np.array(alpha)

# def noisy_measure(x, dt, sigma_W):
# 	x_hat = sigma_W / np.sqrt(dt) * np.random.normal(size=x.shape)
# 	return x_hat

def noisy_measure(x, sigma_W):
	x_hat = x * (1 + sigma_W * np.random.normal(size=x.shape))
	return x_hat

class SensoryFeedback:
	def __init__(self, env, callback_list: dict, step_skip: int, muscle_activation_time=0.0, ramp_up_time=0.0):
		self.env = env
		self.callback_list = callback_list
		self.every = step_skip
		self.s = env.arm_param['s']
		self.ds = self.s[1]-self.s[0]
		self.ctrl_mag = np.zeros([3, env.arm_param['n_elem']+1])
		self.muscle_activation_time = muscle_activation_time
		assert ramp_up_time >= 0.0
		if ramp_up_time == 0:
			self.ramp_up_time = 1e-14
		else:
			self.ramp_up_time = ramp_up_time
		self.count = 0

	def cal_arm_state(self, kappa, ds):
		# theta = np.hstack([0, cum_integral(kappa[1:], ds)])
		# theta = np.hstack([0, cum_integral(kappa[:-1], ds)])
		theta = np.hstack([0, cum_integral(_aver(kappa), ds)])
		pos = np.hstack([np.zeros(2).reshape(2,1), cum_integral(_aver(np.vstack([np.cos(theta), np.sin(theta)])), ds)])
		return pos, theta

	def cal_true_sensory_info(self, kappa, ds, target):
		pos, theta = self.cal_arm_state(kappa, ds)
		target_vector = target[:,None] - pos
		dist = np.sqrt(np.einsum('in,in->n', target_vector, target_vector))
		normalized_target_vector = target_vector / (dist + 1e-16)
		tangent = np.vstack([np.cos(theta), np.sin(theta)])
		normal = np.vstack([-np.sin(theta), np.cos(theta)])
		sin = np.einsum('in,in->n', normalized_target_vector, normal)
		cos = np.einsum('in,in->n', normalized_target_vector, tangent)
		alpha = np.arctan2(sin, cos)
		alpha = np.unwrap(alpha)
		return dist, alpha, theta

	def get_closest_pt(self, conc):
		# softidx = softargmax(conc, self.s[::self.env.sensor_skip])
		# self.max_idx = int(softidx / self.ds)
		self.max_idx = np.argmax(conc) * self.env.sensor_skip
		return self.s[self.max_idx]

	def smooth_alpha(self, dists, alphas, kappa, s):
		ds = s[1] - s[0]
		n_sensor = len(dists)
		gap = int(100/n_sensor)
		res = []
		ss = s[:gap]
		for i in range(n_sensor):
			y0 = np.array([dists[i], alphas[i]])
			kk = kappa[int(i*gap):int((i+1)*gap)]
			# sol = odeint(f, y0, ss, args=(kk,ds))[:,1]
			sol = solve(f, y0, ss, kk, ds)
			res.append(sol)
		res = np.array(res).flatten()
		return res

	def sensor(self, system, target):
		self.env.diffusion_simulator.simulate(target, self.env.time_step)
		r0 = system.position_collection[:2, ::self.env.sensor_skip].T
		
		# sin = system.director_collection[0,1,:]
		# cos = system.director_collection[1,1,:]
		# theta0 = np.hstack([0, np.arctan2(sin, cos)])[::self.env.sensor_skip]

		conc = self.env.diffusion_simulator.get_sensor_conc(r0)
		kappa0 = -system.kappa[0, :]
		kappa = np.hstack([kappa0[0], kappa0, kappa0[-1]])
		self.dist_true, self.alpha_true, self.theta_true = self.cal_true_sensory_info(kappa, self.ds, target)
		# conc += noisy_measure(conc, self.env.time_step, sigma_W =0.0002) # 1e-5 ## noisy concentration measurements
		# kappa += noisy_measure(kappa, self.env.time_step, sigma_W=0.001) ## noisy curvature measurements
		sigma_W = 0. # 0.05 # 0.01
		conc = noisy_measure(conc, sigma_W=sigma_W)
		kappa = noisy_measure(kappa, sigma_W=sigma_W)
		self.env.sensors.simulate(r0, conc, kappa[::self.env.sensor_skip], self.env.time_step) # old
		# self.env.sensors.simulate(conc, kappa[::self.env.sensor_skip], self.env.time_step) # new algorithm
		
		# ## get alpha with true theta instead of theta from proprioception
		# self.env.sensors.theta = self.theta_true[::self.env.sensor_skip]
		# self.env.sensors.alpha = self.env.sensors.beta - self.env.sensors.theta
		# self.env.sensors.callback_list['alpha'][-1] = self.env.sensors.alpha.copy()
		# self.env.sensors.callback_list['theta'][-1] = self.env.sensors.theta.copy()
		
		### Alpha Interpolation between suckers
		# self.alpha_estimate = np.interp(self.s, self.s[::self.env.sensor_skip], self.env.sensors.alpha)
		self.alpha_estimate = np.hstack([self.smooth_alpha(self.dist_true[::self.env.sensor_skip][:-1], self.env.sensors.alpha[:-1], kappa, self.s), self.env.sensors.alpha[-1]])
		self.sin_alpha = np.sin(self.alpha_estimate)
		self.s0 = self.get_closest_pt(conc)

	def LM_choice(self, array):
		return np.where(array>=0), np.where(array<0)
	
	def feedback(self, time, system):
		# mag_cos = system.director_collection.copy()[1,1,:]
		# mag_sin = system.director_collection.copy()[2,1,:]
		error_feedback = self.sin_alpha # mag_cos # mag_cos - mag_sin # 0.5 * (np.sqrt(3)*mag_cos - mag_sin) # 
		idx_top, idx_bottom = self.LM_choice(error_feedback)
		sigma = 0.01 # 0.01
		steep = 300 # 300 # 200
		shift = 1.5 # 1.5 # 2.5
		## Ramp up the muscle torque
		factor = min(1.0, (time - self.muscle_activation_time) / self.ramp_up_time)
		if self.s0 > self.s[10]:
			## longitudinal
			self.ctrl_mag[0,idx_top] = (-1 / (1 + np.exp(-steep * (self.s[idx_top] - (self.s0-shift*sigma)))) + 1) # gaussian(s, s0-4*sigma, sigma) * mag # 2.5
			self.ctrl_mag[1,idx_bottom] = (-1 / (1 + np.exp(-steep * (self.s[idx_bottom] - (self.s0-shift*sigma)))) + 1) # gaussian(s, s0+2.5*sigma, sigma) * mag
		else:
			self.ctrl_mag[0,idx_top] = 1.
			self.ctrl_mag[1,idx_bottom] = 1.
		# mag_feedback = _aver_kernel(abs(error_feedback))
		# mag_feedback[0] = mag_feedback[1]
		# mag_feedback[1] = 0.5 * (mag_feedback[0] + mag_feedback[2])
		mag_feedback = abs(error_feedback)
		self.ctrl_mag[:-1, :] *= factor * mag_feedback # * mag_ramp
		self.ctrl_mag = np.clip(self.ctrl_mag, 0, 1)
	
	def sensory_feedback_law(self, time, system, target):
		self.sensor(system, target)
		if time >= self.muscle_activation_time:
			self.feedback(time, system)
		self.callback()
		self.count += 1
		return self.ctrl_mag
	
	def callback(self):
		if self.count % self.every == 0:
			self.callback_list['dist_true'].append(self.dist_true.copy())
			self.callback_list['alpha_true'].append(self.alpha_true.copy())
			self.callback_list['theta_true'].append(self.theta_true.copy())
			self.callback_list['s_bar'].append(self.max_idx)
			self.callback_list['alpha_hat'].append(self.alpha_estimate.copy())