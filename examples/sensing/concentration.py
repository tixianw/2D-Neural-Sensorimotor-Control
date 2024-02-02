"""
Created on Feb Sep 26, 2023
@author: tixianw2
"""
import numpy as np
from elastica import *

folder_name = 'Data/'
file_name = 'journal_video_sensing_bend_low_c_resolution'
data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()
arm = data['arm']
sensor = data['sensor']
sensor_skip = data['model']['sensor']['sensor_skip']
target_t = data['model']['target']
target = target_t[0]
position = arm[-1]['position'][:,:2,:]
r0 = position[0, :, ::sensor_skip].T
dist_true = np.linalg.norm(r0 - target, axis=1)
mu_true = data['model']['diffusion']['mu_true']
conc_t = sensor[-1]['conc']
conc_bad = conc_t[-1, :]
conc_good = -1/mu_true * np.log(dist_true)
# print(conc_good, conc_bad)
# quit()

class Concentration:
	def __init__(self, bounds, ds, target, mu, callback_list: dict, step_skip: int):
		self.bounds = bounds # [xlb,xub,ylb,yub]
		self.ds = ds
		x = np.linspace(bounds[0], bounds[1], int((bounds[1]-bounds[0])/ds)+1)
		y = np.linspace(bounds[2], bounds[3], int((bounds[3]-bounds[2])/ds)+1)
		self.xy = np.meshgrid(y,x) # [X,Y]
		self.mu = mu
		self.callback_list = callback_list
		self.every = step_skip
		self.D = 1.
		self.intensity = 2*np.pi * self.D / self.mu
		## initialization
		self.get_target(target)
		self.c = np.zeros_like(self.xy[0]) # self.conc.copy() # 
		self.step = 0
	
	def get_target(self, target):
		self.target = target
		self.idx_target = ((target - np.array([self.bounds[0], self.bounds[2]])) / self.ds).astype('int')
		## get saturation at boundary
		dist = np.sqrt((self.xy[0]-target[0])**2 + (self.xy[1]-target[1])**2)
		self.conc = -1/self.mu * np.log(dist)

	def Laplacian(self, u): # Delta_u
		u_pad = np.pad(u, ((1, 1), (1, 1)), mode='edge') # constant(zero-B.C.)
		u_sum = u_pad[:-2,1:-1] + u_pad[2:,1:-1] + u_pad[1:-1,:-2] + u_pad[1:-1,2:]
		Delta_u = (u_sum - 4*u) / self.ds**2
		return Delta_u

	def saturate(self, u):
		for i in range(len(u)):
			if u[0,i] > self.conc[0,i]:
				u[0,i] = self.conc[0,i]
			if u[-1,i] > self.conc[-1,i]:
				u[-1,i] = self.conc[-1,i]
			if u[i,0] > self.conc[i,0]:
				u[i,0] = self.conc[i,0]
			if u[i,-1] > self.conc[i,-1]:
				u[i,-1] = self.conc[i,-1]
		return u
	
	def dynamics(self, u , dt): # u_t = Delta_u + f
		du_dt = self.D * self.Laplacian(u)
		u_next = u + du_dt * dt
		u_next[self.idx_target[1], self.idx_target[0]] = self.intensity
		u_next = self.saturate(u_next) # saturate at the source
		return u_next
	
	def simulate(self, target, dt):
		self.get_target(target)
		self.c = self.dynamics(self.c, dt)
		self.callback()
		self.step += 1
	
	def get_sensor_conc(self, r0):
		idx_sensor = ((r0 - np.array([self.bounds[0], self.bounds[2]])) / self.ds).astype('int')
		conc = self.c[idx_sensor[:,1], idx_sensor[:,0]] ## dynamic concentration
		# conc = self.conc[idx_sensor[:,1], idx_sensor[:,0]] ## static concentration
		conc = conc / conc_bad * conc_good
		return conc

	def callback(self):
		if self.step % self.every == 0:
			self.callback_list['c_map'].append(self.c.copy())