"""
Created on Feb Sep 26, 2023
@author: tixianw2
"""
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from elastica import *
from set_environment import Environment
from neuralctrl_Matsuoka import NeuralCtrl
from sensoryfeedback import SensoryFeedback

def data_save(env, controller, sensor=None, desired=None):
	model = {
		'flags': env.flags,
		'numerics': env.numeric_param,
		'arm': env.arm_param,
		'sensor': env.sensor_param,
		'diffusion': env.diffusion_param,
		'muscle_activation_time': sensor.muscle_activation_time,
		'neuron': controller.neuron_param,
		'sensor_mag': controller.mu,
	}
	if env.flags[1]:
		model.update({'target': env.target[::env.step_skip, :]})
	if env.flags[2]:
		model.update({'obstacle': env.obstacle_param})
	arm = []
	muscle = []
	neuron = []
	sensor_data = []
	arm.append({
		'position': np.array(env.pp_list['position'])[:,:,:],
		'orientation': np.array(env.pp_list['orientation'])[:,:,:,:],
		'kappa': -np.array(env.pp_list['kappa'])[:,0,:] / np.array(env.pp_list['voronoi_dilatation'])[:,:],
		'velocity': np.array(env.pp_list['velocity'])[:,:2,:],
		'omega': np.array(env.pp_list['angular_velocity'])[:,0,:],
		'nu1': np.array(env.pp_list['strain'])[:, -1, :] + 1, 
		'nu2': np.array(env.pp_list['strain'])[:, 1, :],
	})
	muscle.append({
		'u': np.array(env.muscle_list['u']),
	})
	neuron.append({
		'I': np.array(controller.callback_list['I']),
		'V': np.array(controller.callback_list['V']),
		})
	if sensor != None:
		sensor_data.append({
			'dist_true': np.array(sensor.callback_list['dist_true']),
			'alpha_true': np.array(sensor.callback_list['alpha_true']),
			'theta_true': np.array(sensor.callback_list['theta_true']),
			's_bar': np.array(sensor.callback_list['s_bar']),
			'concentration_map': np.array(env.diffusion_list['c_map']),
			'alpha': np.array(env.sensor_list['alpha']),
			'mu': np.array(env.sensor_list['mu']),
			'theta': np.array(env.sensor_list['theta']),
			'target_belief': np.array(env.sensor_list['target_belief']),
			'conc': np.array(env.sensor_list['conc']),
			'alpha_hat': np.array(sensor.callback_list['alpha_hat']),
			'curv': np.array(env.sensor_list['curv'])
		})

	data = {'t': np.array(env.pp_list['time']),
		'model': model,
		'arm': arm,
		'muscle': muscle,
		'neuron': neuron,
		'sensor': sensor_data,
		'desired': desired,
		}
	if not env.flags[0]:
		pass
	else:
		file_name = 'journal_sensorimotor'
	np.save('Data/'+file_name+'.npy', data)

def get_activation(systems, time, controller=None, desired_activation=None, bendpoint=None):
	if controller==None:
		activation = np.zeros([3, systems[0].n_elems+1])
	else:
		activation = controller.neural_ctrl(time, systems[0], desired_activation, bendpoint)
	return activation


def main():
	## Create arm and simulation environment
	final_time = 2.
	flag_shooting = 1
	flag_target = True # False # 
	flag_obstacle = False # True # 
	flags = [flag_shooting, flag_target, flag_obstacle]

	env = Environment(final_time, flags)
	total_steps, systems = env.reset()

	### Create neural muscular controller
	neural_list = defaultdict(list)
	sensor_list = defaultdict(list)
	controller = NeuralCtrl(env, neural_list, env.step_skip)
	muscle_activation_time = 0.0
	sensor = SensoryFeedback(env, sensor_list, env.step_skip, muscle_activation_time=muscle_activation_time)
	
	## Start the simulation
	print("Running simulation ...")
	time = np.float64(0.0)
	for k_sim in tqdm(range(total_steps)):
		target = env.target[k_sim, :]
		u, sbar = sensor.sensory_feedback_law(time, systems[0], target)
		activation = get_activation(systems, time, controller=controller, desired_activation=u, bendpoint=sbar)
		time, systems, done = env.step(time, activation)
		if not k_sim % env.step_skip:
			dist = norm(env.sensors.target_belief.mean(axis=0) - target) / env.arm_param['L']
		if done or dist < 0.005:
			break
		
	data_save(env, controller, sensor)
	return


if __name__ == "__main__":
	main()
