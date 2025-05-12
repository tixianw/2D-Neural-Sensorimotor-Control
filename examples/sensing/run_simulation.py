"""
Created on Feb Sep 26, 2023
@author: tixianw2
"""
import sys
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from elastica import *
from set_environment import Environment
from sensoryfeedback import SensoryFeedback

def data_save(env, controller):
	model = {
		'flags': env.flags,
		'numerics': env.numeric_param,
		'arm': env.arm_param,
		'sensor': env.sensor_param,
		'diffusion': env.diffusion_param,
		'muscle_activation_time': controller.muscle_activation_time,
	}
	if env.flags[1]:
		model.update({'target': env.target[::env.step_skip, :]})
	if env.flags[2]:
		model.update({'obstacle': env.obstacle_param})
	arm = []
	muscle = []
	sensor = []
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
	sensor.append({
		'dist_true': np.array(controller.callback_list['dist_true']),
		'alpha_true': np.array(controller.callback_list['alpha_true']),
		'theta_true': np.array(controller.callback_list['theta_true']),
		's_bar': np.array(controller.callback_list['s_bar']),
		'concentration_map': np.array(env.diffusion_list['c_map']),
		'alpha': np.array(env.sensor_list['alpha']),
		'mu': np.array(env.sensor_list['mu']),
		'theta': np.array(env.sensor_list['theta']),
		'target_belief': np.array(env.sensor_list['target_belief']),
		'conc': np.array(env.sensor_list['conc']),
		'alpha_hat': np.array(controller.callback_list['alpha_hat']),
		'curv': np.array(env.sensor_list['curv'])
		})

	data = {'t': np.array(env.pp_list['time']),
		'model': model,
		'arm': arm,
		'muscle': muscle,
		'sensor': sensor,
		}
	if not env.flags[0]:
		file_name = 'journal_sensing_straight'
	else:
		file_name = 'journal_sensing_bend'
	np.save('Data/'+file_name+'.npy', data)

def get_activation(systems, time, controller=None, target=None):
	if controller==None:
		activation = np.zeros([3,systems[0].n_elems+1])
	else:
		activation = controller.sensory_feedback_law(time, systems[0], target)
	return activation

def get_target(flag_target, flag_shooting):
	if flag_target:
		if not flag_shooting:
			x_star = 0.16
			y_star = 0.16
		else:
			x_star = 0.12
			y_star = 0.10
		target = np.array([x_star, y_star])
		print('target:', target)
		return target
	else:
		return None
	
import click

@click.command()
@click.option(
    "--case",
    type=click.Choice(
        ['straight', 'bend'],
        case_sensitive=False,
    ),
    default='bend',
)

def main(case, target_idx=None):
	## Create arm and simulation environment
	final_time = 2.
	if case == 'straight':
		flag_shooting = 0
	elif case == 'bend':
		flag_shooting = 1
	flag_target = True # False # 
	flag_obstacle = False # True # 
	flags = [flag_shooting, flag_target, flag_obstacle] 

	env = Environment(final_time, flags)
	target_init = get_target(flag_target, flag_shooting)
	env.get_target(target_init)
	
	total_steps, systems = env.reset()

	## Create sensory feedback controller
	sensory_ctrl_list = defaultdict(list)
	muscle_activation_time = final_time+1e-5
	controller = SensoryFeedback(env, sensory_ctrl_list, env.step_skip, muscle_activation_time=muscle_activation_time)

	## Start the simulation
	print("Running simulation ...")
	time = np.float64(0.0)
	for k_sim in tqdm(range(total_steps)):
		target = env.target[k_sim, :]
		activation = get_activation(systems, time, controller, target)
		time, systems, done = env.step(time, activation)
		## make arm fixed
		if flag_shooting:
			systems[0].position_collection = env.init_data['pos'].copy()
			systems[0].kappa = env.init_data['rest_kappa'].copy()
	
	data_save(env, controller)
	return


if __name__ == "__main__":
	main()
