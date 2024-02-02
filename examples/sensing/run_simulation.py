"""
Created on Feb Sep 26, 2023
@author: tixianw2
"""
import sys
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from elastica import *
from set_arm_environment import ArmEnvironment
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
		# 'target_list': env.new_targets,
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
	# np.save('Data/sensing_paper_compare1.npy', data)
	if not env.flags[0]:
		# file_name = 'sensory_test1_'+str(env.sensor_param['n_sensor'])+'sensor_1'
		# file_name = 'sensory_test1_theta1'
		# file_name = 'paper_sensing1'
		# file_name = 'journal_sensing_stat' + str(env.new_targets[1])
		# file_name = 'journal_sensing_straight1' # 1_new_mu2'
		# file_name = 'new_algorithm_test3_1'
		file_name = 'journal_video_sensing_straight'
	else:
		# file_name = 'sensory_test2_'+str(env.sensor_param['n_sensor'])+'sensor_1_smooth'
		# file_name = 'sensory_test2_theta1'
		# file_name = 'paper_sensing2'
		# file_name = 'journal_sensing2_stat' + str(env.new_targets[1])
		# file_name = 'journal_'+str(env.sensor_param['n_sensor'])+'sensor2_smooth'
		# file_name = 'new_algorithm_test4_2'
		# file_name = 'sensorimotor_ctrl5_4'
		# file_name = 'journal_sensing_bend_static_conc2_2' # 2_new_mu5'
		# file_name = 'journal_sensing_noise4' # 2_new'
		# file_name = 'test_asymptotic_stability3'
		# file_name = 'test_sensing_bend'
		file_name = 'journal_video_sensing_bend_local'
	np.save('Data/'+file_name+'.npy', data)
	# np.save('Data/Stats/'+file_name+'.npy', data)

def get_activation(systems, time, controller=None, target=None):
	if controller==None:
		activation = np.zeros([3,systems[0].n_elems+1])
	else:
		activation = controller.sensory_feedback_law(time, systems[0], target)
	return activation

def get_target(flag_target, flag_shooting):
	if flag_target:
		if not flag_shooting:
			x_star = 0.16 # 0.1
			y_star = 0.16 # 0.1
		else:
			x_star = 0.12 # 0.12 #
			y_star = 0.10 # 0.10 #
		target = np.array([x_star, y_star])
		print('target:', target)
		return target
	else:
		return None

def main(target_idx=None):
	## Create arm and simulation environment
	final_time = 2. # 20. # 
	flag_shooting = 1
	flag_target = True # False # 
	flag_obstacle = False # True # 
	flags = [flag_shooting, flag_target, flag_obstacle] 

	# env = ArmEnvironment(final_time, flags)
	env = Environment(final_time, flags)
	target_init = get_target(flag_target, flag_shooting)
	env.get_target(target_init)
	
	total_steps, systems = env.reset()

	## Create sensory feedback controller
	sensory_ctrl_list = defaultdict(list)
	muscle_activation_time = final_time+1e-5 # 0. # 0.5 # 
	controller = SensoryFeedback(env, sensory_ctrl_list, env.step_skip, muscle_activation_time=muscle_activation_time)
	
	# ### automatic set targets
	# L = env.arm_param['L']
	# if flag_shooting:
	# 	target_list = [np.array([0.1*i*L-0.5*L, 0.1*j*L]) for i in range(16) for j in range(11)]
	# else:
	# 	target_list = [np.array([0.1*i*L, 0.1*j*L]) for i in range(11) for j in range(11)]
	# env.new_targets = [target_list, target_idx]
	# print('actual target: ', target_list[int(target_idx)])
	# # print(target_list, len(target_list))
	# # quit()

	## Start the simulation
	print("Running simulation ...")
	time = np.float64(0.0)
	for k_sim in tqdm(range(total_steps)):
		# env.target[k_sim, :] = target_list[int(target_idx)]
		target = env.target[k_sim, :]
		activation = get_activation(systems, time, controller, target)
		time, systems, done = env.step(time, activation)
		## make arm fixed
		if flag_shooting:
			systems[0].position_collection = env.init_data['pos'].copy()
			systems[0].kappa = env.init_data['rest_kappa'].copy()
		# # if k_sim < int(muscle_activation_time / env.time_step):
		# # 	time += env.time_step
		# # else:
		# # 	# print('At ', time, 's, the rod starts moving')
		# # 	time, systems, done = env.step(time, activation)
		# if not k_sim % env.step_skip:
		# 	dist = norm(env.sensors.target_belief.mean(axis=0) - target) / env.arm_param['L']
		# if done or (dist < 0.005 and time > 2.):
		# 	break
	
	data_save(env, controller)
	return


if __name__ == "__main__":
	# import argparse
	# parser = argparse.ArgumentParser(
	# 	description='Run simulation'
	# )
	# parser.add_argument(
	# 	'--filename', type=str, default='simulation',
	# 	help='a str: data file name',
	# )
	# args = parser.parse_args()
	# main(filename=args.filename)

	# main(sys.argv[1])
	main()
