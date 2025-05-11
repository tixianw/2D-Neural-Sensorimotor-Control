"""
Created on Feb Sep 26, 2023
@author: tixianw2
"""
import sys
sys.path.append("../../")
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec 
import numpy as np
from numpy.linalg import norm
from scipy.signal import butter, filtfilt
np.seterr(divide='ignore', invalid='ignore')
from tools import _aver, softargmax
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, 
							   FormatStrFormatter, 
							   AutoMinorLocator)
from bearing_sensing import Cost, Cost_prop, Dist

color = ['C'+str(i%10) for i in range(200)]
marker = ['*', 'x', 's', '>', 'o', '+', '1', 'd'] * 20
[label_size, tick_size, legend_size] = [20, 15, 12]
[solid, dash, dot] = [3, 2, 1] # line_width

def isninf(a):
	return np.all(np.isfinite(a))

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

def main(case):

	folder_name = 'Data/'

	if not os.path.exists(folder_name):
		raise FileNotFoundError(f"Folder not found: {folder_name}, run the simulation first.")
	
	if case == 'straight':
		file_name = 'journal_sensing_straight'
	elif case == 'bend':
		file_name = 'journal_sensing_bend'
		
	data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

	n_elem = data['model']['arm']['n_elem']
	L = data['model']['arm']['L']
	radius = data['model']['arm']['radius']
	base_radius = data['model']['arm']['base_radius']
	if radius[0]==radius[1]:
		r = (radius / radius[0])**1 * 50
	else:
		r = (radius / radius[0])**1 * 200
	dt = data['model']['numerics']['step_size']
	save_step_skip = data['model']['numerics']['step_skip']
	t_t = data['t'] # [:-1]
	final_time = t_t[-1] # data['model']['numerics']['final_time']
	s = data['model']['arm']['s']
	s_mean = (s[1:] + s[:-1])/2
	ds = s[1] - s[0]
	arm = data['arm']
	muscle = data['muscle']
	sensor = data['sensor']
	flags = data['model']['flags']
	flag_target = flags[1]
	flag_obs = flags[2]
	if flag_target:
		target_t = data['model']['target']
	if flag_obs:
		Obs = data['model']['obstacle']
		N_obs = Obs['N_obs']
		print(N_obs, 'obstacles')
		pos_obs = Obs['pos_obs']
		r_obs = Obs['r_obs']
		len_obs = Obs['len_obs']
	## params
	[X,Y] = data['model']['diffusion']['xy']
	[xlb,xub,ylb,yub] = data['model']['diffusion']['bounds']
	[x, y] = [X[0,:], Y[:,0]]
	mu_true = data['model']['diffusion']['mu_true']
	print('mu=', mu_true)
	delta_s = data['model']['sensor']['delta_s']
	n_sensor = data['model']['sensor']['n_sensor']
	sensor_skip = data['model']['sensor']['sensor_skip']
	position = arm[-1]['position'][:,:2,:]
	orientation = arm[-1]['orientation'][:,1:,:-1,:]
	kappa = arm[-1]['kappa']
	u_t = muscle[-1]['u']

	dist_true_t = sensor[-1]['dist_true']
	alpha_true_t = sensor[-1]['alpha_true']
	theta_true_t = sensor[-1]['theta_true']
	s_bar_idx = sensor[-1]['s_bar']
	c_map_t = sensor[-1]['concentration_map']
	alpha_t = sensor[-1]['alpha']
	mu_t = sensor[-1]['mu']
	theta_t = sensor[-1]['theta']
	r_t = sensor[-1]['target_belief']
	conc_t = sensor[-1]['conc']
	curv_t = sensor[-1]['curv']

	target = target_t[0]
	dist = [Dist(r_t, target, L)]
	print('dist', dist[0])

	print('total_steps: ', len(t_t), ', final_time=', final_time, ', target=', target_t[0,...])
	print('k_theta', data['model']['sensor']['gain_theta'], 'n_sensor', n_sensor)


	video = 2
	save_flag = 0

	if video == 2:
		fig = plt.figure(figsize=(24*0.75, 10*0.75))
		gs = gridspec.GridSpec(1, 3, width_ratios=[0.6, 0.2, 0.2], height_ratios=[1])
		gs2 = gridspec.GridSpec(3, 3, width_ratios=[0.5, 0.2, 0.2], height_ratios=[1,1,1])
		ax0 = fig.add_subplot(gs[0])
		ax11 = fig.add_subplot(gs2[1])
		ax12 = fig.add_subplot(gs2[4])
		ax13 = fig.add_subplot(gs2[7])
		ax21 = fig.add_subplot(gs2[2])
		ax22 = fig.add_subplot(gs2[5])
		ax23 = fig.add_subplot(gs2[8])
		var11 = np.sin(alpha_t)
		var12 = theta_t
		var13 = u_t
		var21 = conc_t
		var22 = dist_true_t
		var23 = kappa
		if save_flag:
			factor = 1
			name = file_name
		else:
			factor = 5
			name = 'trash'
		fps = 20 # 10

		video_folder_name = 'Videos/'
		if not os.path.exists(video_folder_name):
			os.makedirs(video_folder_name)
			print(f"Folder didn't exist, created: {video_folder_name}")

		video_name = video_folder_name + name + ".mov"
		FFMpegWriter = manimation.writers["ffmpeg"]
		metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
		writer = FFMpegWriter(fps=fps, metadata=metadata)
		with writer.saving(fig, video_name, 100):
			for jj in range(int((len(t_t)-1) / factor)+1): # +1
				i = jj * factor
				time = i / (len(t_t)-1) * final_time
				plt.clf()
				gs = gridspec.GridSpec(1, 3, width_ratios=[0.6, 0.2, 0.2], height_ratios=[1])
				gs2 = gridspec.GridSpec(3, 3, width_ratios=[0.5, 0.2, 0.2], height_ratios=[1,1,1])
				ax0 = fig.add_subplot(gs[0])
				ax11 = fig.add_subplot(gs2[1])
				ax12 = fig.add_subplot(gs2[4])
				ax13 = fig.add_subplot(gs2[7])
				ax21 = fig.add_subplot(gs2[2])
				ax22 = fig.add_subplot(gs2[5])
				ax23 = fig.add_subplot(gs2[8])
				ax11.cla()
				ax12.cla()
				ax13.cla()
				ax21.cla()
				ax22.cla()
				ax23.cla()
				contourf = ax0.contourf(X,Y,c_map_t[i],50)
				cbar = fig.colorbar(contourf, ax=ax0)
				# cbar.set_clim(0., 1.)
				ax0.text(0.05,0.85, 't: %.3f s'%(time), color='white', transform=ax0.transAxes, fontsize=15, verticalalignment='top')
				if save_flag:
					ax0.text(0.05,0.8, '(%.1fx speed)'%(fps/100), color='white', transform=ax0.transAxes, fontsize=12, verticalalignment='top')
				ax0.scatter(position[i,0,:],position[i,1,:], s=r, marker='o', color='#F0F8FF', alpha=1)
				ax0.scatter(position[i,0,s_bar_idx[i]],position[i,1,s_bar_idx[i]], s=r[s_bar_idx[i]]*0.2, marker='o', color='red', alpha=1)
				for n in range(n_sensor):
					r0 = position[i,:,::sensor_skip].T
					if n==0:
						ax0.scatter(r0[n,0], r0[n,1], s=20, marker='o', color='k', label='sensors')
						ax0.scatter(r_t[i,n,0], r_t[i,n,1], s=20, marker='x', color='C1', label='estimated target')
					else:
						ax0.scatter(r0[n,0], r0[n,1], s=20, marker='o', color='k')
						ax0.scatter(r_t[i,n,0], r_t[i,n,1], s=20, marker='x', color='C1')
				if flag_obs:
					alpha0 = 0.8
					name_obstacle = locals()
					for o in range(N_obs):
						name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
						ax0.add_artist(name_obstacle['obstacle'+str(o)])
				ax0.scatter(target_t[i,0], target_t[i,1], s=50, marker='*', color='C3', label='target point')
				ax0.legend(bbox_to_anchor=(0.01, 0.99), loc=2, ncol=1, borderaxespad=0)
				ax11.plot(s[::sensor_skip], var11[i,:])
				ax11.plot(s, np.sin(alpha_true_t[i,:]), ls='--')
				ax12.plot(s[::sensor_skip], var12[i,:])
				ax12.plot(s, theta_true_t[i,:], ls='--')
				for j in range(var13.shape[1]):
					ax13.plot(s, var13[i,j,:])
				ax21.plot(s[::sensor_skip], var21[i,:])
				ax22.plot(s, var22[i,:])
				ax23.plot(s[1:-1], var23[i,:])
				ax0.axis([xlb, xub, ylb, yub])
				ax0.set_aspect('equal', adjustable='box')
				ax0.set_title('concentration map + target locolization')
				ax11.set_ylabel('$\sin(\\alpha(s,t))$')
				ax12.set_ylabel('$\\theta(s,t)$')
				ax13.set_ylabel('$u^m(s,t)$')
				ax21.set_ylabel('sensor concentration')
				ax22.set_ylabel('target distance')
				ax23.set_ylabel('$\\kappa(s,t)$')
				ax13.set_xlabel('s')
				ax23.set_xlabel('s')
				if not save_flag:
					plt.pause(0.001)
				else:
					writer.grab_frame()
				# break
				if not isninf(position):
					break  


	plt.show()


if __name__ == "__main__":
	main()