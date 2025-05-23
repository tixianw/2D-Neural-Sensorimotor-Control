import os
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
from elastica import *
from set_environment import Environment
from neuralctrl_Matsuoka import NeuralCtrl
# from neuralctrl_FitzHugh_Nagumo import NeuralCtrl
from sensoryfeedback import SensoryFeedback
from tools import gaussian

s = np.linspace(0, 0.2, 101)

def data_save(env, controller, sensor=None, desired=None):
	model = {
		'flags': env.flags,
		'numerics': env.numeric_param,
		'arm': env.arm_param,
		'neuron': controller.neuron_param,
		'sensor_mag': controller.mu,
	}
	if env.flags[1]:
		model.update({'target': env.target[::env.step_skip, :]})
	if env.flags[2]:
		model.update({'obstacle': env.obstacle_param})
	position = []
	arm = []
	muscle = []
	neuron = []
	sensor_data = []
	position.append(np.array(env.pp_list['position'])[:,:2,:])
	arm.append({
		'orientation': np.array(env.pp_list['orientation'])[:,:,:,:],
		'velocity': np.array(env.pp_list['velocity'])[:,:2,:],
		'omega': np.array(env.pp_list['angular_velocity'])[:,0,:],
		'kappa': -np.array(env.pp_list['kappa'])[:,0,:] / np.array(env.pp_list['voronoi_dilatation'])[:,:],
		'nu1': np.array(env.pp_list['strain'])[:, -1, :] + 1, 
		'nu2': np.array(env.pp_list['strain'])[:, 1, :],
		# 'e': np.array(env.pp_list['dilatation'])[:,:],
		# 'v_e': np.array(env.pp_list['voronoi_dilatation'])[:, :],
		# 'g': np.array(env.pp_list['shear_force'])[:, :],
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
		'dist': np.array(sensor.callback_list['dist']),
		'angle': np.array(sensor.callback_list['angle']),
		's_bar': np.array(sensor.callback_list['s_bar']),
		})

	data = {'t': np.array(env.pp_list['time']),
		'model': model,
		'position': position,
		'arm': arm,
		'muscle': muscle,
		'neuron': neuron,
		'sensor': sensor_data,
		'desired': desired,
		}
	
	folder_name = 'Data/'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
		print(f"Directory {folder_name} created")
	
	if env.flags[3]	== 'LM_reach':
		np.save(folder_name + 'journal_neural_reaching_LM.npy', data)
	elif env.flags[3] == 'LM_point':
		np.save(folder_name + 'journal_neural_pointing_LM.npy', data)
	if env.flags[3] == 'TM_reach':
		np.save(folder_name + 'journal_neural_reaching_TM.npy', data)

def get_activation(time, systems, controller=None, desired_activation=None, bendpoint=None):
	activation = controller.neural_ctrl(time, systems[0], desired_activation, bendpoint)
	# activation = np.zeros([3,systems[0].n_elems+1])
	return activation

import click

@click.command()
@click.option(
    "--case",
    type=click.Choice(
        ['LM_reach', 'LM_point', 'TM_reach'],
        case_sensitive=False,
    ),
    default='TM_reach',
)

def main(case):
	### Create arm and simulation environment
	final_time = 4.
	flag_shooting = 1
	flag_target = True # False # 
	flag_obstacle = False # True # 
	flags = [flag_shooting, flag_target, flag_obstacle, case]

	env = Environment(final_time, flags)
	total_steps, systems = env.reset()

	### Create neural muscular controller
	neural_list = defaultdict(list)
	sensor_list = defaultdict(list)
	controller = NeuralCtrl(env, neural_list, env.step_skip)
	sensor = SensoryFeedback(env, sensor_list, env.step_skip)

	### Desired muscle activation or curvature
	V = np.vstack([gaussian(s, mu=0.1, sigma=0.02, magnitude=80), np.zeros([2, len(s)])])
	u = controller.v_to_u(V)
	desired = {
		'desired_u': u,
	}
	
	### Start the simulation
	print("Running simulation ...")
	time = np.float64(0.0)
	for k_sim in tqdm(range(total_steps)):
		target = env.target[k_sim, :]
		u, sbar = sensor.sensory_feedback_law(time, systems[0], target)
		activation = get_activation(
			time, systems, controller=controller, 
			desired_activation=u,
			bendpoint=sbar,
		)
		time, systems, done = env.step(time, activation)
		if done:
			break
	
	data_save(env, controller, sensor, desired)
	return


if __name__ == "__main__":
	main()
