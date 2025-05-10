"""
Created on Fri April 7, 2023
@author: tixianw2
"""
import sys
sys.path.append("../../")
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
from bearing_sensing import Cost, Cost_prop

color = ['C'+str(i%10) for i in range(200)]
marker = ['*', 'x', 's', '>', 'o', '+', '1', 'd'] * 20
[label_size, tick_size, legend_size] = [20, 15, 12]
[solid, dash, dot] = [3, 2, 1] # line_width

# # SMALL_SIZE = 24 # 18
# # MEDIUM_SIZE = 28 # 24
# # BIGGER_SIZE = 36 # 28

# SMALL_SIZE = 36
# MEDIUM_SIZE = 14
# BIGGER_SIZE = 16

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# color = list(mcolors.TABLEAU_COLORS.values())
# linewidth = 3
# mpl.rcParams['axes.linewidth'] = 3 #set the value globally

# mpl.rcParams['xtick.major.size'] = 8
# mpl.rcParams['xtick.major.width'] = 3
# mpl.rcParams['ytick.major.size'] = 8
# mpl.rcParams['ytick.major.width'] = 3

def isninf(a):
	return np.all(np.isfinite(a))

folder_name = 'Data/'
choice = 1
if choice==0:
	file_name = 'test3'
elif choice==1:
	# file_name = 'journal_sensing_bend'
	# file_name = 'journal_sensorimotor_global1'
	file_name = 'journal_sensorimotor7' # _noise'
	
data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

n_elem = data['model']['arm']['n_elem']
L = data['model']['arm']['L']
radius = data['model']['arm']['radius']
base_radius = data['model']['arm']['base_radius']
if radius[0]==radius[1]:
	r = (radius / radius[0])**1 * 50
else:
	r = (radius / radius[0])**1 * 200
final_time = data['model']['numerics']['final_time']
dt = data['model']['numerics']['step_size']
save_step_skip = data['model']['numerics']['step_skip']
t_t = data['t']
s = data['model']['arm']['s']
s_mean = (s[1:] + s[:-1])/2
ds = s[1] - s[0]
arm = data['arm']
muscle = data['muscle']
sensor = data['sensor']
neuron = data['neuron']
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
delta_s = data['model']['sensor']['delta_s']
n_sensor = data['model']['sensor']['n_sensor']
sensor_skip = data['model']['sensor']['sensor_skip']
position = arm[-1]['position'][:,:2,:]
orientation = arm[-1]['orientation'][:,1:,:-1,:]
kappa = arm[-1]['kappa']
# velocity = arm[-1]['velocity']
# vel_mag = np.sqrt(np.einsum('ijn,ijn->in', velocity, velocity))
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
alpha_hat_t = sensor[-1]['alpha_hat']
V_t = neuron[-1]['V']
I_t = neuron[-1]['I']

# print(s_bar_idx)
# print(np.argmin(dist_true_t, axis=-1))
# print(np.argmax(conc_t, axis=-1))
# soft = []
# for i in range(len(t_t)):
#     soft.append(softargmax(conc_t[i], s[::sensor_skip]))
# print(soft)
# quit()

print('total_steps: ', len(t_t), ', final_time=', final_time, 'target', target_t[0])
print('k_theta', data['model']['sensor']['gain_theta'], 'n_sensor', n_sensor)

plot = 0
save_plot = 0
if plot:
	# plt.figure()
	# j = 0
	# for j in range(len(V_t)):
	#     plt.plot(s, V_t[j, :, :].T)

	cost = Cost(r_t, mu_t, n_sensor)
	costp = Cost_prop(theta_t, curv_t, n_sensor, delta_s)
	print(cost[-1], costp[-1])
	costp[costp<1e-16] = 1e-16
	fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(15,10))
	axes[0].semilogy(t_t, cost, lw=5)
	axes[1].semilogy(t_t, costp, lw=5)
	# if save_plot:
	plt.tight_layout()

	# cost = Cost(r_t, mu_t, n_sensor)
	# print(cost[-1])
	# plt.figure(figsize=(9,6))
	# plt.semilogy(t_t, cost)
	# plt.xlabel('t')
	# plt.ylabel('Cost')
	# plt.xlabel('$t$', fontsize=label_size)
	# plt.ylabel('$\\epsilon(t)$', fontsize=label_size)
	# plt.xticks(fontsize=tick_size)
	# plt.yticks(fontsize=tick_size)
	# if save_plot:
	#     plt.tight_layout()
	#     plt.savefig('Figures/consensus_cost.png')

	plt.figure(figsize=(9,6)) # (16*0.7,9*0.7)
	for i in range(n_sensor):
		plt.plot(t_t, alpha_t[:,i], ls='-', lw=solid, color=color[i]) #, label='$\hat{\\alpha}^{%d}(t)$'%(i+1))
		plt.plot(t_t, alpha_true_t[:,::sensor_skip][:,i], ls=':', lw=dash, color='k') #, label='$\\alpha^{%d}(t)$'%(i+1))
		# plt.axhline(alpha[i], color='k', ls=':', lw=dot)
	plt.xlabel('$t$', fontsize=label_size)
	plt.ylabel('$\\alpha(t)$', fontsize=label_size)
	# plt.legend(bbox_to_anchor=(1.0,1.0), loc='lower right', ncol=5, borderaxespad=0.1, fontsize=legend_size)
	plt.xticks(fontsize=tick_size)
	plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
		['$-\\pi$', '$-\\pi/2$','0', '$\\pi/2$', '$\\pi$'], fontsize=tick_size)
	plt.xlim(t_t[0], t_t[-1])
	plt.ylim(-np.pi, np.pi)
	if save_plot:
		plt.tight_layout()
		plt.savefig('Figures/consensus_alpha.png')
	
	plt.figure(figsize=(9,6)) # (16*0.7,9*0.7)
	for i in range(n_sensor):
		plt.plot(t_t, theta_t[:,i], ls='-', lw=solid, color=color[i]) #, label='$\hat{\\alpha}^{%d}(t)$'%(i+1))
		plt.plot(t_t, theta_true_t[:,::sensor_skip][:,i], ls=':', lw=dash, color='k') #, label='$\\alpha^{%d}(t)$'%(i+1))
	plt.xlabel('$t$', fontsize=label_size)
	plt.ylabel('$\\theta(t)$', fontsize=label_size)
	# plt.legend(bbox_to_anchor=(1.0,1.0), loc='lower right', ncol=5, borderaxespad=0.1, fontsize=legend_size)
	plt.xticks(fontsize=tick_size)
	plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
		['$-\\pi$', '$-\\pi/2$','0', '$\\pi/2$', '$\\pi$'], fontsize=tick_size)
	plt.xlim(t_t[0], t_t[-1])
	plt.ylim(-np.pi, np.pi)

	plt.figure(figsize=(9,6)) # (16*0.7,9*0.7)
	for i in range(n_sensor):
		plt.plot(t_t, mu_t[:,i], ls='-', lw=solid, color=color[i]) #, label='$\hat{\\mu}^{%d}(t)$'%(i+1))
	plt.axhline(mu_true, ls=':', lw=dash, color='k') #, label='$\\mu(t)$')
	plt.xlabel('$t$', fontsize=label_size)
	plt.ylabel('$\\mu(t)$', fontsize=label_size)
	# plt.legend(bbox_to_anchor=(1.0,1.0), loc='lower right', ncol=5, borderaxespad=0.1, fontsize=legend_size)
	plt.xticks(fontsize=tick_size)
	plt.yticks(fontsize=tick_size)
	# plt.yticks([0, 0.5, 1., 1.5, 2.], ['0', '0.5', '1.0', '1.5', '2.0'], fontsize=tick_size)
	plt.xlim(t_t[0], t_t[-1])
	if save_plot:
		plt.tight_layout()
		plt.savefig('Figures/consensus_mu.png')

	plt.figure(figsize=(9,6))
	plt.plot(position[-1,0,:], position[-1,1,:])
	plt.scatter(target_t[-1,0], target_t[-1,1], s=200, marker='*', color='C3', label='target')
	for i in range(n_sensor):
		plt.plot([position[-1,0,::sensor_skip][i], r_t[-1,i,0]], [position[-1,1,::sensor_skip][i], r_t[-1,i,1]], ls='--', lw=dot, color=color[i]) #, label='sensor$^{%d}$'%(i+1))
		plt.scatter(r_t[-1,:,0], r_t[-1,:,1], s=200, marker='x', color='C2')
	plt.legend(bbox_to_anchor=(1.0,1.0), loc='lower right', ncol=5, borderaxespad=0.1, fontsize=legend_size)
	plt.xlabel('$x$', fontsize=label_size)
	plt.ylabel('$y$', fontsize=label_size)
	plt.xticks(fontsize=tick_size)
	plt.yticks(fontsize=tick_size)
	if save_plot:
		plt.tight_layout()
		plt.savefig('Figures/consensus_triangle.png')

	plt.figure(figsize=(9,6))
	for i in range(n_sensor):
		plt.plot(r_t[:,i,0], r_t[:,i,1], ls='--', lw=dash, color=color[i]) #, label='sensor$^{%d}$'%(i+1))
	plt.plot(target_t[:,0], target_t[:,1], ls=':', lw=dash, color='C3')
	plt.scatter(target_t[-1,0], target_t[-1,1], s=200, marker='*', color='C3', label='target')
	plt.scatter(r_t[-1,:,0], r_t[-1,:,1], s=200, marker='x', color='C2')
	plt.legend(bbox_to_anchor=(1.0,1.0), loc='lower right', ncol=5, borderaxespad=0.1, fontsize=legend_size)
	plt.xlabel('$x$', fontsize=label_size)
	plt.ylabel('$y$', fontsize=label_size)
	plt.xticks(fontsize=tick_size)
	plt.yticks(fontsize=tick_size)
	if save_plot:
		plt.tight_layout()
		plt.savefig('Figures/consensus_agents.png')

	plt.show()
	# quit()

if choice==0:
	video = 2 # 1
elif choice==1:
	video = 2 # 2 # 3
save_flag = 0

## arm and concentration
if video == 1:
	fig = plt.figure(figsize=(16*0.8, 16*0.8))
	ax0 = fig.add_subplot(1, 1, 1)
	if save_flag:
		factor = 1
		name = file_name
	else:
		factor = 5 # int(2000 / save_step_skip)
		name = 'trash'
	fps = 10
	video_name = 'Videos/' + name + ".mov"
	FFMpegWriter = manimation.writers["ffmpeg"]
	metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
	writer = FFMpegWriter(fps=fps, metadata=metadata)
	with writer.saving(fig, video_name, 100):
		for jj in range(int((len(t_t)-1) / factor)+1): # +1
			i = jj * factor
			time = i / (len(t_t)-1) * final_time
			plt.clf()
			ax0 = fig.add_subplot(1, 1, 1)
			contourf = ax0.contourf(X,Y,c_map_t[i],50)
			contourf.cmap.set_over('red')
			contourf.cmap.set_under('blue')
			contourf.changed()
			cbar = fig.colorbar(contourf, ax=ax0)
			# cbar.set_clim(0., 1.)
			ax0.text(0.05,0.85, 't: %.3f s'%(time), color='white', transform=ax0.transAxes, fontsize=15, verticalalignment='top')
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
			ax0.scatter(target_t[i,0], target_t[i,1], s=50, marker='*', color='C3', label='target point')
			if flag_obs:
				alpha0 = 0.8
				name_obstacle = locals()
				for o in range(N_obs):
					name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
					ax0.add_artist(name_obstacle['obstacle'+str(o)])
			ax0.legend(bbox_to_anchor=(0.01, 0.99), loc=2, ncol=1, borderaxespad=0)
			ax0.axis([xlb, xub, ylb, yub])
			ax0.set_aspect('equal', adjustable='box')
			ax0.set_title('concentration map + target locolization')
			if not save_flag:
				plt.pause(0.001)
			else:
				writer.grab_frame()
			# break
			if not isninf(position):
				break

elif video == 2:
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
		factor = 5 # int(2000 / save_step_skip)
		name = 'trash'
	fps = 20 # 10
	video_name = 'Videos/' + name + ".mov"
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
			theta = var12[i,:] % (2*np.pi)
			ax12.plot(s[::sensor_skip], theta)
			ax12.plot(s, theta_true_t[i,:], ls='--')
			ax12.set_ylim([-2*np.pi, 2*np.pi])
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
