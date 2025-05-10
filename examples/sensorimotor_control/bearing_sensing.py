"""
Created on Wed Sep 20, 2023
@author: tixianw2
"""
import numpy as np
from numpy.linalg import norm
from elastica import *
from tools import _aver

def Cost(r, mu, n):
	cost = 0
	for i in range(n):
		for j in [max(0,i-1), min(n-1,i+1)]:
			cost += norm(r[..., i, :] - r[..., j, :], axis=-1)**2 + (mu[..., i] - mu[..., j])**2
	return cost / 2 / (n-1)

def Cost_prop(theta, kappa, n, ds):
	cost = n-1
	curv = _aver(kappa) # kappa # 
	for i in range(n-1):
		cost -= np.cos(theta[..., i+1] - theta[..., i] - curv[..., i] * ds) # curv[..., i+1]
	return cost / (n-1) / 2

def Dist(r, target, L):
	# aver_r = r.mean(axis=1)
	# dist = norm(aver_r - target[None,:], axis=1) / L
	dists = norm(r - target, axis=-1) / L
	dist = dists.mean(axis=-1)
	# print(np.amin(dist))
	return np.amin(dist) # dist[-1] # 

class BearingSensing:
	def __init__(self, delta_s, n_sensor, alpha0, mu0, theta0, gain_mu, gain_r, gain_theta, \
	      callback_list: dict, step_skip: int, neurons_chemo=None, neurons_proprio=None):
		self.delta_s = delta_s
		self.n_sensor = n_sensor
		self.gain_mu = gain_mu
		self.gain_r = gain_r
		self.gain_theta = gain_theta
		self.callback_list = callback_list
		self.every = step_skip
		self.neurons_chemo = neurons_chemo
		self.neurons_proprio = neurons_proprio
		if neurons_chemo is None:
			self.alpha = alpha0
			self.theta = theta0
		else:
			self.alpha = neurons_chemo.angle
			self.theta = neurons_proprio.angle
		
		self.beta = self.alpha + self.theta
		self.mu = mu0
		self.Laplacian_matrix()
		self.step = 0
	
	def init(self, conc):
		self.pos_local(conc)
	
	def adjacency_matrix(self):
		## neighbor connected
		adj_mat = np.zeros([self.n_sensor, self.n_sensor])
		adj_mat[0,1] = 1
		for i in range(1,self.n_sensor-1):
			adj_mat[i][i-1] = 1
			adj_mat[i][i+1] = 1
		adj_mat[-1,-2] = 1
		return adj_mat
	
	def degree_matrix(self):
		## neighbor connected
		deg_mat = np.diag([1]+[2 for i in range(self.n_sensor-2)]+[1])
		return deg_mat
	
	def Laplacian_matrix(self):
		self.Laplacian = self.degree_matrix() - self.adjacency_matrix()
	
	def R(self, angle):
		matrix = np.vstack([np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]).T
		return matrix.reshape(self.n_sensor, 2, 2)
	
	def pos_local(self, conc):
		self.conc = conc
		self.r = np.vstack([np.cos(self.beta), np.sin(self.beta)]).T * self.rho(conc)[:,None]
	
	def rho(self, conc):
		dist = np.exp(-self.mu * conc)
		return dist

	def sensor_pos_diff(self, kappa):
		array = np.vstack([np.sin(self.theta), -np.cos(self.theta)]).T * kappa[:,None] * self.delta_s**2
		array[0] = -np.array([np.cos(self.theta[1]), np.sin(self.theta[1])]) * self.delta_s
		array[-1] = np.array([np.cos(self.theta[-1]), np.sin(self.theta[-1])]) * self.delta_s
		return array
	
	def chemosensing(self, conc, kappa, r0, dtheta_dt):
		dr_dt = -self.gain_r * (self.Laplacian @ self.r + self.sensor_pos_diff(kappa))
		# dr_dt = -self.gain_r * (self.Laplacian @ (self.r + r0))
		# print(np.sum((self.Laplacian @ r0 - self.sensor_pos_diff(kappa, theta0))**2))
		temp = np.einsum('nij,nj->ni', self.R(-self.beta), dr_dt)
		dalpha_dt = self.rho(conc) * temp[:,1] - dtheta_dt
		dmu_dt = -(conc * self.rho(conc)) * temp[:,0] - self.Laplacian @ self.mu * self.gain_mu
		return dalpha_dt, dmu_dt

	def proprioception(self, kappa):
		# theta_tilde = self.theta[:-1] + _aver(kappa) * self.delta_s # kappa[1:]
		# dtheta_dt = -self.gain_theta * np.sin(self.theta[1:] - theta_tilde)
		# return np.hstack([0, dtheta_dt])
		theta_tilde1 = self.theta[2:] - _aver(kappa)[1:] * self.delta_s # kappa[1:-1] # kappa[2:]
		theta_tilde2 = self.theta[1:-1] - _aver(kappa)[:-1] * self.delta_s # kappa[:-2] # kappa[1:-1]
		dtheta_dt = self.gain_theta * (np.sin(theta_tilde1 - self.theta[1:-1]) - np.sin(theta_tilde2 - self.theta[:-2]))
		dthetaN_dt = -self.gain_theta* np.sin(self.theta[-1] - self.theta[-2] - _aver(kappa)[-1] * self.delta_s) # kappa[-2] # kappa[-1]
		return np.hstack([0, dtheta_dt, dthetaN_dt])
	
	def simulate(self, r0, conc, kappa, dt):
		dtheta_dt = self.proprioception(kappa)
		dalpha_dt, dmu_dt = self.chemosensing(conc, kappa, r0, dtheta_dt)
		if self.neurons_chemo == None:
			self.alpha += dalpha_dt * dt
			'''
			neuron rings generate ~1e-6 time step size
			without neuron rings, dt=1e-5 makes system not stable
			should use Runge-Kutta method
			'''
			self.theta += dtheta_dt * dt
		else:
			self.neurons_chemo.simulate(dalpha_dt)
			self.neurons_proprio.simulate(dtheta_dt)
			# print((self.neurons_chemo.angle - self.alpha)/dalpha_dt)
			self.alpha = self.neurons_chemo.angle
			self.theta = self.neurons_proprio.angle
		# self.alpha = np.unwrap(self.alpha)
		self.theta = np.unwrap(self.theta)
		self.beta = self.alpha + self.theta
		self.mu += dmu_dt * dt
		self.pos_local(conc)
		self.kappa = kappa
		self.target_belief = r0 + self.r
		self.callback()
		self.step += 1

	def callback(self):
		if self.step % self.every == 0:
			self.callback_list['alpha'].append(self.alpha.copy())
			self.callback_list['mu'].append(self.mu.copy())
			self.callback_list['theta'].append(self.theta.copy())
			self.callback_list['target_belief'].append(self.target_belief.copy())
			self.callback_list['conc'].append(self.conc.copy())
			self.callback_list['curv'].append(self.kappa.copy())