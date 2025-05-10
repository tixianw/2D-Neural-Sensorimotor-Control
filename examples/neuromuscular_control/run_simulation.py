import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
from elastica import *
from set_arm_environment import ArmEnvironment
from set_environment import Environment
# from neuralctrl import NeuralCtrl
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
	# name = 'tau'+str(controller.neuron_param['tau'])+\
	# 	'_lmd'+str(controller.neuron_param['lmd'])+\
	# 	'_tauA'+str(controller.neuron_param['tau_adapt'])+\
	# 	'_inhibit'+str(controller.neuron_param['inhibition'])+\
	# 	'_adapt'+str(controller.neuron_param['adaptation'])+\
	# 	'_V'+str(controller.neuron_param['V0'][0])+'-'+str(controller.neuron_param['V0'][1])+'-'+str(controller.neuron_param['V0'][2])+'-'+str(controller.neuron_param['V0'][3])
	# np.save('Data/paper_bend_'+name+'.npy', data)
	# np.save('Data/paper_backstepping.npy', data)
	# np.save('Data/test_init1_3.npy', data)
	# np.save('Data/test_backstepping2_gamma'+str(controller.gamma)+'_tracking2.npy', data)
	# np.save('Data/journal_neural_transverse1.npy', data)
	# np.save('Data/test_neural_lmd'+str(controller.neuron_param['lmd'])+'_mu'+str(controller.mu)+'_2.npy', data)
	np.save('Data/journal_neural_reaching_TM.npy', data) # reaching # pointing_LM
	### PID
	# pid = controller.PID_param
	# print('PID:', pid)
	# name = 'order'+str(pid[3])+'_P'+str(pid[0])+'_I'+str(pid[1])+'_D'+str(pid[2])
	# np.save('Data/test_ES_'+name+'.npy', data)
	### Sensory feedback
	# mu = controller.mu
	# # inhibit = controller.neuron_param['inhibition']
	# adapt = controller.neuron_param['adaptation']
	# print('mu=', mu, 'adapt=', adapt)
	# name = '_adapt'+str(adapt)+'_'+str(controller.neuron_param['V0'][0])+'-'+str(controller.neuron_param['V0'][2])
	# if env.flags[0]:
	# 	np.save('Data/paper_sensor_mu'+str(mu)+name+'.npy', data)
	# else:
	# 	pass
	# 	# np.save('Data/paper_sensor_mu'+str(mu)+name+'.npy', data)

	# name = 'tau'+str(controller.neuron_param['tau'])+\
	# 	'_lmd'+str(controller.neuron_param['lmd'])+\
	# 	'_inhibit'+str(controller.neuron_param['inhibition'])+\
	# 	'_V'+str(controller.neuron_param['V0'][0])+'-'+str(controller.neuron_param['V0'][1])
	# np.save('Data/FN_test_'+name+'.npy', data)
	# mu = controller.mu
	# inhibit = controller.neuron_param['inhibition']
	# print('mu=', mu, 'inhibit=', inhibit)
	# name = '_inhibit'+str(inhibit)
	# if env.flags[0]:
	# 	np.save('Data/FN_test_sensor2_mu'+str(mu)+name+'.npy', data)
	# else:
	# 	np.save('Data/FN_test_sensor_mu'+str(mu)+name+'.npy', data)

	# np.save('Data/test_wave1.npy', data)
	# np.save('Data/test_inhibition2_adaptation2_20_I10_tau0.03.npy', data)
	# np.save('Data/test_inhibition2_adaptation2_20_I13_10.npy', data)
	# np.save('Data/test_sensor2_1.npy', data)
	# np.save('Data/test_PID0_P10.npy', data)
	# np.save('Data/test_PID_P20I20.npy', data)
	# np.save('Data/test_PID_shoot_P15I10D0.05.npy', data)
	# np.save('Data/ES_PID_P40I15D0.06.npy', data)
	# np.save('Data/ES_PID_shoot_P40I15.npy', data)

def get_activation(time, systems, controller=None, desired_curvature=None, desired_activation=None, bendpoint=None):
	activation = controller.neural_ctrl(time, systems[0], desired_curvature, desired_activation, bendpoint)
	# activation = np.zeros([3,systems[0].n_elems+1])
	return activation


def main(filename):
	### Create arm and simulation environment
	final_time = 4. # 6.0 # 4.0 # 1.001
	flag_shooting = 1
	flag_target = True # False # 
	flag_obstacle = False # True # 
	flags = [flag_shooting, flag_target, flag_obstacle]

	# env = ArmEnvironment(final_time, flags)
	env = Environment(final_time, flags)
	total_steps, systems = env.reset()

	### Create neural muscular controller
	neural_list = defaultdict(list)
	sensor_list = defaultdict(list)
	controller = NeuralCtrl(env, neural_list, env.step_skip)
	sensor = SensoryFeedback(env, sensor_list, env.step_skip)

	### Desired muscle activation or curvature
	# u = np.vstack([gaussian(s, mu=0.1, sigma=0.02, magnitude=0.2), np.zeros([2, len(s)])])
	V = np.vstack([gaussian(s, mu=0.1, sigma=0.02, magnitude=80), np.zeros([2, len(s)])])
	u = controller.v_to_u(V)
	# desired_kappa = gaussian(s[1:-1], mu=0.1, sigma=0.02, magnitude=20)
	desired = {
		'desired_u': u,
		# 'desired_kappa': desired_kappa, 
		# 'desired_pos': desired_position,
	}
	# # plt.figure()
	# # plt.plot(s[1:-1], desired_kappa)
	# # plt.plot(s[1:-1], -systems[0].kappa[0,:])
	# # plt.figure()
	# # plt.plot(desired_position[0,:], desired_position[1,:])
	# # plt.show()
	# # quit()
	
	### Start the simulation
	print("Running simulation ...")
	time = np.float64(0.0)
	for k_sim in tqdm(range(total_steps)):
		target = env.target[k_sim, :]
		u, sbar = sensor.sensory_feedback_law(time, systems[0], target)
		activation = get_activation(
			time, systems, controller=controller, 
			# desired_curvature=desired_kappa,
			desired_activation=u,
			bendpoint=sbar,
		)
		time, systems, done = env.step(time, activation)
		if done:
			break
	
	data_save(env, controller, sensor, desired)
	return


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		description='Run simulation'
	)
	parser.add_argument(
		'--filename', type=str, default='simulation',
		help='a str: data file name',
	)
	args = parser.parse_args()
	main(filename=args.filename)
