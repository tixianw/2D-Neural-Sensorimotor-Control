import numpy as np
from elastica import *
from set_arm_environment import *
from tools import target_traj, d2r, _aver, cum_integral
from concentration import Concentration
from bearing_sensing import BearingSensing

def cal_arm_state(kappa, ds):
	theta = np.hstack([0, cum_integral(_aver(kappa), ds)])
	pos = np.hstack([np.zeros(2).reshape(2,1), cum_integral(_aver(np.vstack([np.cos(theta), np.sin(theta)])), ds)])
	return pos, theta

def cal_true_sensory_info(kappa, ds, target):
	pos, theta = cal_arm_state(kappa, ds)
	target_vector = target[:,None] - pos
	dist = np.sqrt(np.einsum('in,in->n', target_vector, target_vector))
	normalized_target_vector = target_vector / (dist + 1e-16)
	tangent = np.vstack([np.cos(theta), np.sin(theta)])
	normal = np.vstack([-np.sin(theta), np.cos(theta)])
	sin = np.einsum('in,in->n', normalized_target_vector, normal)
	cos = np.einsum('in,in->n', normalized_target_vector, tangent)
	alpha = np.arctan2(sin, cos)
	return dist, alpha, theta

def alpha_mu_initialization(kappa, ds, target, sensor_skip, L, mu_true, theta0):
	pos, theta = cal_arm_state(kappa, ds)
	dist_true, alpha_true, theta_true = cal_true_sensory_info(kappa, ds, target)
	# alpha_true = np.unwrap(alpha_true)
	r0 = pos[:, ::sensor_skip]
	# theta0 = theta[::sensor_skip]
	offset = np.array([L*2.5, L*1.2]) # ([L*2.5, L*0.8]) # ([L*1.5, L*0.8]) # 
	target_beliefs = np.random.normal(size=r0.shape) * 0.1 + (target + offset)[:,None]
	target_vector = target_beliefs - r0
	dist = np.sqrt(np.einsum('in,in->n', target_vector, target_vector))
	normalized_target_vector = target_vector / (dist + 1e-16)
	tangent = np.vstack([np.cos(theta0), np.sin(theta0)])
	normal = np.vstack([-np.sin(theta0), np.cos(theta0)])
	sin = np.einsum('in,in->n', normalized_target_vector, normal)
	cos = np.einsum('in,in->n', normalized_target_vector, tangent)
	alpha = np.arctan2(sin, cos)
	# alpha = np.unwrap(alpha)
	conc = -1/mu_true * np.log(dist_true[::sensor_skip])
	mu = -1/conc * np.log(dist)
	# plt.figure()
	# plt.scatter(target[0], target[1], s=200, marker='*')
	# plt.scatter(target_beliefs[0,:], target_beliefs[1,:], s=50, marker='.')
	# plt.axis([offset[0]-0.2,target[0]+0.2,target[1]-0.2,offset[1]+0.2])
	# print(mu)
	# plt.figure()
	# s = np.linspace(0, L, len(theta))
	# plt.plot(s, alpha_true)
	# plt.scatter(s[::sensor_skip], alpha)
	# plt.show()
	# quit()
	return [alpha, mu]

# Add call backs
class CallBack(CallBackBaseClass):
	"""
	Call back function for the arm
	"""
	def __init__(self, step_skip: int, callback_params: dict, total_steps: int):
		CallBackBaseClass.__init__(self)
		self.every = step_skip
		self.callback_params = callback_params
		self.total_steps = total_steps
		self.count = 0

	def make_callback(self, system, time, current_step: int):
		self.callback_params['potential_energy'].append(
			system.compute_bending_energy() + system.compute_shear_energy()
		)
		if self.count % self.every == 0: # and self.count != self.total_steps:
			self.callback_params["time"].append(time)
			# self.callback_params["step"].append(current_step)
			self.callback_params["position"].append(
				system.position_collection.copy()
			)
			self.callback_params["orientation"].append(
				system.director_collection.copy()
			)
			self.callback_params["velocity"].append(
				system.velocity_collection.copy()
			)
			self.callback_params["angular_velocity"].append(
				system.omega_collection.copy()
			)
			
			self.callback_params["kappa"].append(
				system.kappa.copy()
			)
			self.callback_params['strain'].append(
				system.sigma.copy() # + np.array([0, 0, 1])[:, None]
			)
			self.callback_params['dilatation'].append(
				system.dilatation.copy()
			)
			self.callback_params['voronoi_dilatation'].append(
				system.voronoi_dilatation.copy()
			)
			
		self.count += 1
		return

class Environment(ArmEnvironment):

	def setup(self):
		## Set up a rod
		self.set_arm()
		## Set up the target
		if self.flag_target:
			self.set_target()
			self.set_diffusion()
		if self.flag_obstacle:
			self.set_obstacle()
		self.set_sensors()
		return
		
	def set_target(self):
		if not self.flag_shooting:
			x_star = 0.16 / 0.2 * self.arm_param['L'] # 0.1
			y_star = 0.16 / 0.2 * self.arm_param['L'] # 0.1
		else:
			x_star = 0.10 / 0.2 * self.arm_param['L'] # 0.12 #
			y_star = 0.12 / 0.2 * self.arm_param['L'] # 0.10 # 
		target0 = np.array([x_star,y_star])
		print('target:', target0)
		self.target = np.zeros([self.total_steps, 2])
		
		# ## moving target
		# pos1 = np.array([x_star, y_star])
		# pos2 = np.array([-x_star*1., y_star])
		# # pos2 = np.array([x_star*1.5, -y_star])
		# # target_traj(target, pos1, pos1, 0., 0.15, self.total_steps)
		# target_traj(self.target, pos1, pos2, 0., 1., self.total_steps)
		# # target_traj(target, pos2, pos2, 0.8, 1., self.total_steps)

		## static target
		self.target[:,:] = target0
	
	def set_diffusion(self):
		L = self.arm_param['L']
		bounds = [-L*2, L*2, -L*2, L*2]
		Delta_s = 0.01
		mu = 2. # 1. # 5.
		self.diffusion_list = defaultdict(list)
		self.diffusion_simulator = Concentration(bounds, Delta_s, self.target[0,:], mu, self.diffusion_list, self.step_skip)
		self.diffusion_param = {
			'bounds': bounds,
			'Delta_s': Delta_s,
			'xy': self.diffusion_simulator.xy,
			'mu_true': mu,
			'D': self.diffusion_simulator.D,
			'intensity': self.diffusion_simulator.intensity,
		}

	def set_sensors(self):
		n_sensor = 51 # 21 ## number of sensors (first one locate at the origin)
		delta_s = self.arm_param['L']/(n_sensor-1) ## distance betweew sensors
		self.sensor_skip = int(self.shearable_rod.n_elems / (n_sensor-1))
		r0 = self.shearable_rod.position_collection[:2, ::self.sensor_skip].T ## location of sensors
		### random initial alpha's and mu's
		np.random.seed(2048) # 1024
		arm_kappa = -self.shearable_rod.kappa[0]
		kappa = np.hstack([arm_kappa[0], arm_kappa, arm_kappa[-1]])
		s = self.arm_param['s']
		# dist_true, alpha_true, theta_true = cal_true_sensory_info(kappa, s[1]-s[0], self.target[0])
		# alpha_true = np.unwrap(alpha_true[:])
		'''
			numpy version: adding argument period=2*np.pi gives error: LLVM ERROR: Incompatible object format!!!
		'''
		# alpha0 = d2r(np.array([np.random.randint(181) for i in range(n_sensor)])) # - theta_true[::self.sensor_skip]
		# # alpha0 = d2r(np.array([np.random.randint(361)-180 for i in range(n_sensor)]) * 0.1) + alpha_true[::self.sensor_skip]
		# # plt.plot(s, alpha_true)
		# # plt.scatter(s[::self.sensor_skip], alpha0)
		# # plt.show()
		# # quit()
		theta0 = np.hstack([0, d2r(np.random.randn(n_sensor-1)*10)]) * 1. # 18 # np.zeros_like(alpha0)
		alpha0, _ = alpha_mu_initialization(kappa, s[1]-s[0], self.target[0], self.sensor_skip, self.arm_param['L'], self.diffusion_param['mu_true'], theta0)
		mu0 = np.array([np.random.uniform(self.diffusion_simulator.mu*0.5, self.diffusion_simulator.mu*1.5) for i in range(n_sensor)]) # np.array([np.random.uniform(0.5*mu, 2*mu) for i in range(n_sensor)])
		gain_mu = 4e4 # 4e4 # 
		gain_r = 4e4 # 4e4 # 
		gain_theta = 5e4 # 1e4 # 5e4 # 1e5 # 
		neurons_chemo = None
		neurons_proprio = None
		self.sensor_list = defaultdict(list)
		self.sensors = BearingSensing(delta_s, n_sensor, alpha0, mu0, theta0, gain_mu, gain_r, gain_theta, self.sensor_list, self.step_skip, neurons_chemo, neurons_proprio)
		conc = self.diffusion_simulator.get_sensor_conc(r0)
		# self.sensors.init(r0, conc) # old
		self.sensors.init(conc) # new algorithm
		self.sensor_param = {
			'delta_s': delta_s,
			'n_sensor': n_sensor,
			'gain_mu': gain_mu,
			'gain_r': gain_r,
			'gain_theta': gain_theta, 
			'sensor_skip': self.sensor_skip,
		}

	# flag_ring = 0 ## whether to use neural rings or not
	# neurons_chemo = None
	# neurons_proprio = None

	def set_obstacle(self):
		N_obs = 2
		r_obs = np.ones([N_obs]) * 0.02 # 0.01 # 0.04
		len_obs = np.ones([N_obs]) * 0.04
		pos_obs = [np.array([0.065,0.05,0]), np.array([0.14,0.05,0])] # [np.array([0.1,0.05,0])] # [np.array([0.,0.03,0]), np.array([0.08,0.03,0])] # [np.hstack([target_exp+0.01,0])] # 
		# [np.array([0.065,0.05,0]), np.array([0.135,0.05,0])] # 
		self.obstacle_param = {
			'N_obs': N_obs, 
			'r_obs': r_obs, 
			'pos_obs': np.array(pos_obs), 
			'len_obs': len_obs,
			}
		name_obstacle = locals()
		for o in range(N_obs):
			name_obstacle['obstacle'+str(o)] = Cylinder(
				start=pos_obs[o],
				direction=self.normal,
				normal=self.direction,
				base_length=len_obs[o],
				base_radius=r_obs[o],
				density=500,
			)
			self.simulator.append(name_obstacle['obstacle'+str(o)])
			self.simulator.constrain(name_obstacle['obstacle'+str(o)]).using(
				OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
			)
			self.simulator.connect(self.shearable_rod, name_obstacle['obstacle'+str(o)]).using(
				ExternalContact, k=1e2, nu=1.0
			)
	
	def callback(self):
		self.pp_list = defaultdict(list)
		self.simulator.collect_diagnostics(self.shearable_rod).using(
			CallBack, step_skip=self.step_skip, 
			callback_params=self.pp_list, total_steps=self.total_steps,
		)

