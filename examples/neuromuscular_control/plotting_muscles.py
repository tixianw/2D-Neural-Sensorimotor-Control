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
from tools import _diff, _aver, _diff_kernel, _aver_kernel
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, 
                               FormatStrFormatter, 
                               AutoMinorLocator)
from tools import gaussian

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

choice = 2
folder_name = 'Data/' 
if choice==0: ### equilibrium, initialization, backstepping, 
    save_frame = 0
    tau = 0.04 # 0.2 # 0.04 #
    list_name = [tau, 0.1, 10*tau, 0., 1., 60, 40, 80, 0] # negative lambda means tapered
    print('target_list', list_name)
    subname = 'tau'+str(list_name[0])+\
		'_lmd'+str(list_name[1])+\
		'_tauA'+str(list_name[2])+\
		'_inhibit'+str(list_name[3])+\
		'_adapt'+str(list_name[4])+\
		'_V'+str(list_name[5])+'-'+str(list_name[6])+'-'+str(list_name[7])+'-'+str(list_name[8])
    # file_name = 'test_'+subname
    # file_name = 'test_init1'
    # file_name = 'test_backstepping2_gamma3_tracking2'
    # file_name = 'paper_curl_'+subname
    # file_name = 'paper_bend_curl_'+subname
    # file_name = 'paper_backstepping1'
elif choice==1: ### PID control
    pid = [6, 10, 0, 0]
    subname = 'order'+str(pid[3])+'_P'+str(pid[0])+'_I'+str(pid[1])+'_D'+str(pid[2])
    file_name = 'test_ES_'+subname
elif choice==2: ### sensory feedback
    mu, adapt, Vt0, Vt1 = 500, 1., 60, 80 # 40, 120 # 
    subname = 'mu'+str(mu)+'_adapt'+str(adapt)+'_'+str(Vt0)+'-'+str(Vt1)
    # file_name = 'paper_sensor_' + subname
    # file_name = 'journal_neural_reaching'
    # file_name = 'journal_neural_transverse2'
    # lmd = 0.02 # 0.1
    # mu = 200 # 500
    # file_name = 'test_neural_lmd'+str(lmd)+'_mu'+str(mu)+'_2'
    file_name = 'journal_neural_reaching_TM' # pointing_LM' # 
elif choice==3:
    # save_frame = 0
    # list_name = [1., -0.5, 4., 40, 10] # negative lambda means tapered
    # print('target_list', list_name)
    # subname = 'tau'+str(list_name[0])+\
	# 	'_lmd'+str(list_name[1])+\
	# 	'_inhibit'+str(list_name[2])+\
	# 	'_V'+str(list_name[3])+'-'+str(list_name[4])
    # file_name = 'FN_test_'+subname
    mu, inhibit = 500, 0.
    subname = 'mu'+str(mu)+'_inhibit'+str(inhibit)
    file_name = 'FN_test_sensor2_' + subname

data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()
# if choice==1:
    # data2 = np.load(folder_name + file_name2 + '.npy', allow_pickle='TRUE').item()
    # data3 = np.load(folder_name + file_name3 + '.npy', allow_pickle='TRUE').item()

n_elem = data['model']['arm']['n_elem']
L = data['model']['arm']['L']
radius = data['model']['arm']['radius']
if radius[0]==radius[1]:
    r = ((2*radius) / (2*radius[0]))**2 * 50
else:
    r = ((2*radius) / (2*radius[0]))**2 * 100
E = data['model']['arm']['E']
final_time = data['model']['numerics']['final_time']
dt = data['model']['numerics']['step_size']
t = data['t']
s = data['model']['arm']['s']
s_mean = (s[1:] + s[:-1])/2
ds = s[1] - s[0]
position = data['position']
arm = data['arm']
muscle = data['muscle']
neuron = data['neuron']
sensor = data['sensor']
s_bar_idx = sensor[-1]['s_bar']
save_step_skip = data['model']['numerics']['step_skip']
flags = data['model']['flags']
flag_target = flags[1]
flag_obs = flags[2]
if flag_target:
    target = data['model']['target']
if flag_obs:
    Obs = data['model']['obstacle']
    N_obs = Obs['N_obs']
    print(N_obs, 'obstacles')
    pos_obs = Obs['pos_obs']
    r_obs = Obs['r_obs']
    len_obs = Obs['len_obs']
if len(arm[-1]['orientation'][0,...])==3:
    orientation = arm[-1]['orientation'][:,1:,:-1,:]
elif len(arm[-1]['orientation'][0,...])==2:
    orientation = arm[-1]['orientation']
else:
    print('error!')
velocity = arm[-1]['velocity']
vel_mag = np.sqrt(np.einsum('ijn,ijn->in', velocity, velocity))
if choice==0:
    desired_u = data['desired']['desired_u']
elif choice==1:
    desired = data['desired']
    desired_kappa = desired['desired_kappa']
    # desired_pos = desired['desired_pos']

# data_init = {
#     'pos': np.vstack([position[0][-1,:,:], np.zeros(n_elem+1)]), 
#     'rest_kappa': np.vstack([-arm[0]['kappa'][-1,:], np.zeros([2, n_elem-1])]),
#     'rest_V': neuron[0]['V'][-1,:,:],
#     'model': data['model'],
# }
# np.save('Data/initial_data_'+str(list_name[4])+'-'+str(list_name[5])+'-'+str(list_name[7])+'.npy', data_init)
# quit()

# file_name0 = 'neuron_paper_compare1' # 'neuron_paper_backstepping_compare'
# data0 = np.load(folder_name + file_name0 + '.npy', allow_pickle='TRUE').item()
# kappa0 = data0['arm'][-1]['kappa'][:, :]
# # file_name1 = 'test_backstepping1_gamma10' # 'paper_backstepping1'
# # data1 = np.load(folder_name + file_name1 + '.npy', allow_pickle='TRUE').item()
# # data_paper = {
# #         't': t, 's':s/0.2, 
# #         'kappa_benchmark': data0['arm'][-1]['kappa'][:, :],
# #         'kappa_backstepping': data1['arm'][-1]['kappa'][:, :],
# #         'kappa_sensory': arm[-1]['kappa'],
# #         'sbar0': data0['sensor'][-1]['s_bar'],
# #         'sbar1': data1['sensor'][-1]['s_bar'],
# #         'sbar2': data['sensor'][-1]['s_bar']
# #     }
# # np.save('Plotting_for_paper/paper_plot_kappa_compare1-2.npy', data_paper)
# # quit()

# # data_compare = {
# #     't': t, 's': s, 'target': target, 'pos': position, 'arm': arm, 'muscle': muscle
# # }
# # np.save('Data/paper_compare.npy', data_compare)
# # # quit()


actual_list = [
    data['model']['neuron']['tau'], data['model']['neuron']['lmd'], 
    data['model']['neuron']['tau_adapt'], data['model']['neuron']['inhibition'], 
    data['model']['neuron']['adaptation'], 
    data['model']['neuron']['V0'][0], data['model']['neuron']['V0'][1], 
    data['model']['neuron']['V0'][2], data['model']['neuron']['V0'][3]]
print('actual_list', actual_list)

print('total_steps: ', len(t), ', final_time=', final_time)
print('target=', target[0,:])


base_radius = data['model']['arm']['base_radius']
A = np.pi * base_radius**2
I = A**2 / (4*np.pi)
EA = E * A
EI = E * I # (I[1:] + I[:-1]) / 2

def low_pass_filter(signal):
    fs = 60 # 80.0 # 100  # sample rate, Hz
    cutoff = 1 # 2 # desired cutoff frequency of the filter[Hz] slightly higher than actual 1.2[Hz] 
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2   # sin wave can be approx represented as quadratic
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, signal)
    return y

def vel_material():
    # orientation = arm[-1]['orientation']
    vector_a = orientation[:,:,1,:]
    vector_b = orientation[:,:,0,:]
    vel_tan = np.einsum('ijn,ijn->in', _aver(velocity), vector_a)
    vel_per = np.einsum('ijn,ijn->in', _aver(velocity), vector_b)
    return vel_tan, vel_per

def Bend_Vel():
    idx_iter = -1
    # kappa = arm[idx_iter]['kappa']
    # orientation = arm[idx_iter]['orientation']
    # vector_a = orientation[:,:,1,:]
    # vector_b = orientation[:,:,0,:]
    dist = _aver(neuron[idx_iter]['dist'])
    position_diff = np.diff(position[idx_iter], axis=-1)
    lengths = norm(position_diff, axis=1)
    tangents = position_diff / lengths[:, None, :]
    # idx_init_bend = np.argmax(kappa[0,:])
    # idx_bend = np.argmax(kappa[:,idx_init_bend:], axis=1)
    idx_min_dist = np.argmin(dist, axis=1)
    bend_velocity = np.zeros(len(velocity))
    # bend_velocity[:] += np.einsum('ni,ni->n', _aver(velocity[:, :, :])[np.arange(len(velocity)),:,idx_min_dist], tangents[np.arange(len(velocity)), :, idx_min_dist])
    # bend_velocity[:] += np.einsum('ni,ni->n', _aver(velocity[:, :, idx_bend+1]), tangents[:, :, idx_bend+1])
    # bend_velocity[:] += norm(velocity[np.arange(len(velocity)), :, idx_bend+1], axis=1)
    bend_velocity = vel_mag[np.arange(len(velocity)), idx_min_dist]
    bend_pos = idx_min_dist.astype('float64')  / n_elem * L
    return bend_velocity, bend_pos

def Bell_Vel():
    idx_iter = -1
    # kappa = arm[idx_iter]['kappa']
    # orientation = arm[idx_iter]['orientation']
    # vector_a = orientation[:,:,1,:]
    # vector_b = orientation[:,:,0,:]
    dist = _aver(neuron[idx_iter]['dist'])
    position_diff = np.diff(position[idx_iter], axis=-1)
    lengths = norm(position_diff, axis=1)
    tangents = position_diff / lengths[:, None, :]
    # idx_init_bend = np.argmax(kappa[0,:])
    # idx_bend = np.argmax(kappa[:,idx_init_bend:], axis=1)
    idx_min_dist = np.argmin(dist, axis=1)
    idx_reach = idx_min_dist[dist[np.arange(len(idx_min_dist)), idx_min_dist]>=0.001]
    bend_velocity = np.zeros(len(idx_reach))
    # bend_velocity[:] += np.einsum('ni,ni->n', _aver(velocity[:, :, :])[np.arange(len(velocity)),:,idx_min_dist], tangents[np.arange(len(velocity)), :, idx_min_dist])
    # bend_velocity[:] += np.einsum('ni,ni->n', _aver(velocity[:, :, idx_bend+1]), tangents[:, :, idx_bend+1])
    # bend_velocity[:] += norm(velocity[np.arange(len(velocity)), :, idx_bend+1], axis=1)
    bend_velocity = vel_mag[np.arange(len(idx_reach)), idx_reach]
    bend_pos = idx_reach.astype('float64')  / n_elem * L
    return bend_velocity, bend_pos

def Bend_Vel_material():
    n_step = len(velocity)
    idx_iter = -1
    kappa = arm[idx_iter]['kappa']
    omega = arm[idx_iter]['omega']
    vector_a = orientation[:,:,1,:]
    vector_b = orientation[:,:,0,:]
    dist = _aver(neuron[idx_iter]['dist'])
    idx_min_dist = np.argmin(dist, axis=1)
    vel_tan = np.einsum('ni,ni->n', _aver(velocity[:, :, :])[np.arange(n_step),:,idx_min_dist], vector_a[np.arange(n_step),:,idx_min_dist])
    vel_per = np.einsum('ni,ni->n', _aver(velocity[:, :, :])[np.arange(n_step),:,idx_min_dist], vector_b[np.arange(n_step),:,idx_min_dist])
    # vel_a = dist[np.arange(n_step), idx_min_dist] * (omega[np.arange(n_step), idx_min_dist] - kappa[np.arange(n_step), idx_min_dist] * vel_tan) / \
    #     (1 - dist[np.arange(n_step), idx_min_dist] * kappa[np.arange(n_step), idx_min_dist])
    # vel_sbar = vel_a - vel_tan # (dist[np.arange(n_step), idx_min_dist]*omega[np.arange(n_step), idx_min_dist] - vel_tan) / \
    #     # (1 - dist[np.arange(n_step), idx_min_dist] * kappa[np.arange(n_step), idx_min_dist])  # 
    return vel_tan, vel_per # , vel_a, vel_sbar

def omega_kappa_v1():
    idx_iter = -1
    kappa = arm[idx_iter]['kappa']
    omega = arm[idx_iter]['omega']
    orientation = arm[idx_iter]['orientation']
    vector_a = orientation[:,:,1,:]
    dist = _aver(neuron[idx_iter]['dist'])
    idx_min_dist = np.argmin(dist, axis=1)
    vel_tan = np.einsum('kin,kin->kn', _aver(velocity), vector_a)
    array = omega - _aver_kernel(kappa) * vel_tan # dist * (omega - _aver_kernel(kappa) * vel_tan) / (1 - dist * _aver_kernel(kappa))
    return array, idx_min_dist

def fn_Gamma():
    n_step = len(velocity)
    idx_iter = -1
    vector_b = orientation[:,:,0,:]
    dist = _aver(neuron[idx_iter]['dist'])
    idx_min_dist = np.argmin(dist, axis=1)
    vel_per = np.einsum('ij,ij->i', _aver(velocity[:, :, :])[np.arange(n_step),:,idx_min_dist], vector_b[np.arange(n_step),:,idx_min_dist])
    Gamma_bar = vel_per / _aver(vel_mag)[np.arange(n_step),idx_min_dist]
    target_vector = (target[...,None] - _aver(position[0]))
    Gamma = np.einsum('ijn,ijn->in', _aver(velocity[:, :, :]), target_vector) / (_aver(vel_mag)+1e-16) / dist
    return Gamma_bar, Gamma

def fn_omega_kappav1():
    idx_iter = -1
    kappa = arm[idx_iter]['kappa']
    omega = arm[idx_iter]['omega']
    vector_a = orientation[:,:,1,:]
    v1 = np.einsum('ijn,ijn->in', _aver(velocity[:, :, :]), vector_a)
    return omega - _aver_kernel(kappa) * v1

def plot_bend_vel():
    bend_velocity, bend_pos = Bell_Vel() # Bend_Vel()
    t_t = t[:len(bend_velocity)]
    vel_tan = abs(bend_velocity)*100
    pos_vel = bend_pos*100
    vel_filtered = low_pass_filter(vel_tan)
    fig = plt.figure(figsize=(16*0.75, 9*0.75))
    ax = fig.add_subplot(111)
    ax.plot(t_t, vel_tan) #, label='bend velocity')
    ax.plot(t_t, vel_filtered, lw=3, label='bend velocity')
    ax.set_xlabel('t (s)',fontsize=15)
    ax.set_ylabel('bend velocity (cm/s)',fontsize=15)
    ax.legend(bbox_to_anchor=(0, 1), loc=3, ncol=2, borderaxespad=0, fontsize=15)
    ax.set_xlim([-0.1,1.5]) # 1.1
    ax2=ax.twinx()
    pos_filtered = low_pass_filter(pos_vel)
    ax2.plot(t_t, pos_vel, color='C2') #, label='bend position')
    ax2.plot(t_t, pos_filtered, color='C3', label='bend position')
    ax2.set_ylabel('bend position (cm)',fontsize=15)
    ax2.set_ylim([0,L*100])
    ax2.legend(bbox_to_anchor=(1, 1), loc=4, ncol=2, borderaxespad=0, fontsize=15)
    if flag_savefig:
        plt.savefig('Figures/bend_velocity1_1.png', dpi=300)

def plot_bend_vel_exp():
    data_vel = np.load('Data/bend_vel_exp1.npy', allow_pickle='TRUE').item()
    t_exp = data_vel['t1']
    bend_vel_exp = data_vel['bend_vel']
    bend_vel_exp_filtered = data_vel['bend_vel_filtered']
    bend_velocity, bend_pos = Bell_Vel() # Bend_Vel()
    bend_velocity = bend_velocity[:-5]
    t_t = t[:len(bend_velocity)]
    bend_vel = abs(bend_velocity)
    vel_filtered = low_pass_filter(bend_vel)
    fig = plt.figure(figsize=(16*0.75, 9*0.75))
    ax = fig.add_subplot(111)
    # ax.plot(t_t, bend_vel, label='simulation')
    ax.plot(t_t, vel_filtered, lw=5, label='simulation')
    # ax.plot(t_exp, bend_vel_exp, label='experiment')
    ax.plot(t_exp, bend_vel_exp_filtered, lw=5, label='experiment')
    # ax.legend()
    ax.set_xlabel('t (s)',fontsize=15)
    ax.set_ylabel('bend velocity (m/s)',fontsize=15)
    # ax.legend(bbox_to_anchor=(0, 1), loc=3, ncol=2, borderaxespad=0, fontsize=15)
    ax.set_xlim([0.,1.5]) # 1.1
    if flag_savefig:
        plt.savefig('Figures/bend_vel_exp.png', dpi=300)
        plt.savefig('Figures/bend_vel_exp.eps', format='eps')

def plot_sbar_vel():
    bend_velocity, bend_pos = Bend_Vel()
    print(dt, ds)
    v_sbar = _diff(_aver_kernel(low_pass_filter(bend_pos))) / dt
    v_sbar[0] = 0
    v_sbar[-1] = 0
    vel_tan, vel_per = Bend_Vel_material() # , vel_a, vel_sbar 
    vel_tan_smooth = low_pass_filter(vel_tan)
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(t, vel_tan)
    ax[0].plot(t, vel_tan_smooth)
    ax[0].plot(t, vel_per)
    # ax[0].plot(t, -v_sbar)
    # ax[1].plot(t, vel_a)
    # ax[1].plot(t, vel_sbar)
    ax[1].plot(t, v_sbar)
    ax[1].plot(t, vel_tan_smooth)
    ax[1].plot(t, v_sbar+vel_tan_smooth)
    ax[1].set_xlabel('t', fontsize=15)
    ax[0].set_ylabel('$v_1,v_2$', fontsize=15)
    ax[1].set_ylabel('$\\bar{v}_1, v_{\\bar{s}}$', fontsize=15)

def plot_Gamma_sbar():
    Gamma_bar, Gamma = fn_Gamma()
    fig, ax = plt.subplots(nrows=1)
    ax.plot(t, Gamma_bar)
    ax.set_ylim([-1.2,1.2])
    ax.set_xlabel('t', fontsize=15)
    ax.set_ylabel('$\\bar{\\Gamma}$', fontsize=15)

def plot_arm():
    min_var_x = -.8*L
    max_var_x = 1.1*L
    min_var_y = -.8*L 
    max_var_y = 1.1*L
    fig = plt.figure(figsize=(10*0.6, 10*0.6))
    ax0 = fig.add_subplot(1, 1, 1)
    if flag_obs:
        alpha0 = 0.8
        name_obstacle = locals()
        for o in range(N_obs):
            name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
            ax0.add_artist(name_obstacle['obstacle'+str(o)])
    i = -1
    ax0.scatter(position[-1][i,0,:],position[-1][i,1,:], s=r, marker='o', alpha=1,zorder=2)
    # ax0.scatter(position[k][i,0,idx],position[k][i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
    ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
    ax0.axis('off')
    if choice==0 and save_frame:
        plt.savefig('Figures/'+file_name+'.png', transparent=True)

plot = 0
flag_savefig = 0
if plot:
    plot_arm()
    # plot_bend_vel()
    # plot_bend_vel_exp()
    # plot_sbar_vel()
    # plot_Gamma_sbar()


if choice==0:
    video = 2 # 1 # 
elif choice==1:
    video = 2 # 2 # 3
elif choice==2:
    video = 2 # 2 # 3
# if save_frame:
#     video = -1
# else:
#     video = 2
save_flag = 0

## No target, only arm
if video == 0:
    max_var = L*1.1
    min_var = -L/2
    idx = -1
    min_var_x = min(np.amin(position[idx][0,0,:])*1.1, np.amin(position[idx][-1,0,:])*1.1, min_var*1.01)
    max_var_x = max(np.amax(position[idx][-1,0,:])*1.1, np.amax(position[idx][0,0,:])*1.1, max_var*1.01)
    min_var_y = min(np.amin(position[idx][-1,1,:])*1.1, min_var*1.01)
    max_var_y = max_var*1.01
    # dist = neuron[idx]['dist'][:,:]
    fig = plt.figure(figsize=(10*0.6, 10*0.6))
    ax0 = fig.add_subplot(1, 1, 1)
    if save_flag:
        factor1 = 5 # min(int(1000 / save_step_skip), 1) # 5
        if choice==0:
            name = file_name # + '_bend_vel'
    else:
        factor1 = int(2000 / save_step_skip)
        name = 'trash'
    fps = 5 # 10
    video_name = 'Videos/' + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        start = len(position)-1 # 0
        for k in range(start, len(position)):
            for jj in range(int((len(t)-1) / factor1)+1): # +1
                i = jj * factor1
                time = i / (len(t)-1) * final_time
                # idx = np.argmin(dist[i, :])
                ax0.cla()
                if flag_obs:
                    alpha0 = 0.8
                    name_obstacle = locals()
                    for o in range(N_obs):
                        name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
                        ax0.add_artist(name_obstacle['obstacle'+str(o)])
                ax0.scatter(position[k][i,0,:],position[k][i,1,:], s=r, marker='o', alpha=1,zorder=2)
                # ax0.scatter(position[k][i,0,idx],position[k][i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
                ax0.text(L*0.1, max_var*1.05, 't: %.3f s'%(time), fontsize=12)
                angle = np.linspace(0, 2*np.pi, 100)
                # distance = norm(target[0,:])
                # ax0.plot(target[0,0]+distance*np.cos(angle), target[0,1]+distance*np.sin(angle), ls='--', color='black')
                ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
                if not save_flag:
                    plt.pause(0.001)
                else:
                    writer.grab_frame()
            # break
            if not isninf(position[k]):
                break

##
elif video == 1:
    max_var = max(L*1.1, np.amax(target[:,0])*1.1) # max(L*1.1, target[0]*1.1) # 1.5
    min_var = min(-L/2, np.amin(target[:,0])*1.1) # /3 # /4
    idx = -1
    min_var_x = min(np.amin(position[idx][0,0,:])*1.1, np.amin(position[idx][-1,0,:])*1.1, min_var*1.01)
    max_var_x = max(np.amax(position[idx][-1,0,:])*1.1, np.amax(position[idx][0,0,:])*1.1, max_var*1.01)
    min_var_y = min(np.amin(position[idx][-1,1,:])*1.1, min_var*1.01)
    max_var_y = max_var*1.01
    dist = neuron[idx]['dist'][:,:]
    fig = plt.figure(figsize=(10*0.6, 10*0.6))
    ax0 = fig.add_subplot(1, 1, 1)
    if save_flag:
        factor1 = 5 # min(int(1000 / save_step_skip), 1) # 5
        if choice==0:
            name = file_name # + '_bend_vel'
    else:
        factor1 = int(2000 / save_step_skip)
        name = 'trash'
    fps = 5 # 10
    video_name = 'Videos/' + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        start = len(position)-1 # 0
        for k in range(start, len(position)):
            for jj in range(int((len(t)-1) / factor1)+1): # +1
                i = jj * factor1
                time = i / (len(t)-1) * final_time
                idx = np.argmin(dist[i, :])
                ax0.cla()
                if flag_obs:
                    alpha0 = 0.8
                    name_obstacle = locals()
                    for o in range(N_obs):
                        name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
                        ax0.add_artist(name_obstacle['obstacle'+str(o)])
                ax0.scatter(position[k][i,0,:],position[k][i,1,:], s=r, marker='o', alpha=1,zorder=2)
                ax0.scatter(position[k][i,0,idx],position[k][i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
                ax0.text(L*0.1, max_var*1.05, 't: %.3f s'%(time), fontsize=12)
                ax0.scatter(target[i,0], target[i,1], s=200, marker='*', label='target point',zorder=1)
                angle = np.linspace(0, 2*np.pi, 100)
                # distance = norm(target[0,:])
                # ax0.plot(target[0,0]+distance*np.cos(angle), target[0,1]+distance*np.sin(angle), ls='--', color='black')
                ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
                if not save_flag:
                    plt.pause(0.001)
                else:
                    writer.grab_frame()
            # break
            if not isninf(position[k]):
                break


## V and I from cable equation
elif video == 2:
    max_var = L*1.1
    min_var = -L/2 
    idx = -1
    min_var_x = -.8*L # min(np.amin(position[idx][0,0,:])*1.1, np.amin(position[idx][-1,0,:])*1.1, min_var*1.01)
    max_var_x = 1.1*L # max(np.amax(position[idx][-1,0,:])*1.1, np.amax(position[idx][0,0,:])*1.1, max_var*1.01)
    min_var_y = -.8*L # min(np.amin(position[idx][-1,1,:])*1.1, min_var*1.01)
    max_var_y = 1.1*L # max_var*1.01
    var1 = neuron[idx]['I'][:,:,:]
    var2 = neuron[idx]['V'][:,:,:]
    var3 = muscle[idx]['u'][:,:,:]
    kappa = arm[idx]['kappa'][:, :]
    min_var1 = []
    max_var1 = []
    min_var2 = []
    max_var2 = []
    min_var3 = []
    max_var3 = []
    for i in range(var1.shape[1]):
        min_var1.append(np.amin(var1[:, i, :]))
        max_var1.append(np.amax(var1[:, i, :]))
        min_var2.append(np.amin(var2[:, i, :]))
        max_var2.append(np.amax(var2[:, i, :]))
        min_var3.append(np.amin(var3[:, i, :]))
        max_var3.append(np.amax(var3[:, i, :]))
    var1_min = -1 # -500 # min(min_var1)*1-max(max_var1)*0.1
    var1_max = 1000 # 500 # max(max_var1)*1.1
    var2_min = min(min_var2)-1 # -20 # 
    var2_max = max(max_var2)+1 # 60 # 
    var3_min = -0.05 # min(min_var3)*1-max(max_var3)*0.1
    var3_max = 1.05 # max(max_var3)*1.1
    min_kappa = -100 # np.amin(kappa) #
    max_kappa = 150 # np.amax(kappa) # 
    fig = plt.figure(figsize=(30*0.6, 9*0.6))
    ax0 = fig.add_subplot(1,3,1)
    ax1 = fig.add_subplot(2,3,2)
    ax2 = fig.add_subplot(2,3,3)
    ax3 = fig.add_subplot(2,3,5)
    ax4 = fig.add_subplot(2,3,6)
    if save_flag:
        factor1 = 1 # 5
        name = file_name # + '_shooting' # 'sensor_shooting' # '_zero_current' # '_backstepping_constant_u' #
    else:
        factor1 = 5 # int(5000 / save_step_skip) # 50 # 
        name = 'trash'
    slow_factor = 2 # 0.5 # 
    fps = 100 / slow_factor
    video_name = 'Videos/' + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        start = len(position)-1 # 0
        for k in range(start, len(position)):
            var1 = neuron[k]['I'][:,:,:]
            var2 = neuron[k]['V'][:,:,:]
            var3 = muscle[k]['u'][:,:,:]
            kappa = arm[k]['kappa'][:,:]
            for jj in range(int((len(t)-1) / factor1)+1): # +1
                i = jj * factor1
                time = i / (len(t)-1) * final_time
                ax0.cla()
                if flag_obs:
                    alpha0 = 0.8
                    name_obstacle = locals()
                    for o in range(N_obs):
                        name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
                        ax0.add_artist(name_obstacle['obstacle'+str(o)])
                ax0.scatter(position[k][i,0,:],position[k][i,1,:], s=r, marker='o', alpha=1, zorder=2)
                # ax0.scatter(position[k][i,0,idx],position[k][i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
                ax0.text(L*0.1, max_var*1.05, 't: %.3f s'%(time), fontsize=12)
                if flag_target:
                    ax0.scatter(target[i,0], target[i,1], s=100, marker='*', label='target point',zorder=1)
                ax1.cla()
                ax2.cla()
                ax3.cla()
                ax4.cla()
                for ii in range(var1.shape[1]):
                    ax1.plot(s, var1[i, ii, :])
                    ax2.plot(s, var2[i, ii, :])
                    ax3.plot(s, var3[i, ii, :])
                    # ax3.plot(s, desired_u[ii, :], ls='--', color='k')
                    if choice==2:
                        ax3.plot(s, var1[i, ii, :]/mu, ls='--', color='k')
                        # pass
                if not choice==2:
                    ax3.plot(s, (var3[i, 0, :]-var3[i, 1, :]), ls='--')
                ax4.plot(s[1:-1], kappa[i,:])
                # ax4.plot(s[1:-1], kappa0[i,:], ls='--')
                if choice==0:
                    pass
                    # desired_u = gaussian(s, mu=0.1, sigma=0.02, magnitude=0.2)
                    # ax3.plot(s, desired_u, ls='--', color='k')
                elif choice==1:
                    pass
                    # desired_kappa = gaussian(s[1:-1], mu=0.1, sigma=0.02, magnitude=20)
                    # ax0.plot(desired_pos[0,:], desired_pos[1,:], ls='--', color='k')
                    # ax4.plot(s[1:-1], desired_kappa, ls='--', color='k')
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax1.set_ylabel('$I_i$', fontsize=12)
                ax2.set_ylabel('$V_i$', fontsize=12)
                ax3.set_ylabel('$u_i$', fontsize=12)
                ax3.set_xlabel('$s$',fontsize=12)
                ax4.set_xlabel('$s$',fontsize=12)
                ax4.set_ylabel('$\\kappa$', fontsize=12)
                ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
                ax1.axis([-0.01, L+0.01, var1_min, var1_max])
                ax2.axis([-0.01, L+0.01, var2_min, var2_max])
                ax3.axis([-0.01, L+0.01, var3_min, var3_max])
                ax4.axis([-0.01, L+0.01, min_kappa, max_kappa])
                # break
                if not save_flag:
                    plt.pause(0.001)
                else:
                    writer.grab_frame()
            if not isninf(position[k]):
                break
            
            # if choice==0 and save_frame:
            #     plt.savefig('Figures/'+file_name+'.png')


## velocity profiles
elif video==22:
    max_var = L * 1.5 # max(L*1.1, np.amax(target[:,0])*1.1) # max(L*1.1, target[0]*1.1) # 1.5
    min_var = -L * 0.5 # min(-L/2, np.amin(target[:,0])*1.1) # /3 # /4
    idx = -1
    min_var_x = min(np.amin(position[idx][0,0,:])*1.1, np.amin(position[idx][-1,0,:])*1.1, min_var*1.01)
    max_var_x = max(np.amax(position[idx][-1,0,:])*1.1, np.amax(position[idx][0,0,:])*1.1, max_var*1.01)
    min_var_y = min(np.amin(position[idx][-1,1,:])*1.1, min_var*1.01)
    max_var_y = max_var*1.01
    vel_tan, vel_per = vel_material()
    kappa = arm[idx]['kappa'][:, :]
    var1 = arm[idx]['omega']
    var2 = kappa * _aver(vel_per)
    var3 = muscle[idx]['u'][:, :, :]
    var4 = vel_tan # neuron[idx]['dist'][:,:]
    var5 = vel_per # neuron[idx]['angle'][:,:]
    # var4 = neuron[idx]['rigidity'][:,:]
    # var5 = external[idx]['muscle_strain']
    # var6 = external[idx]['force_length_fn']
    # var4 = external[idx]['external_forces'][:,0,:]
    # var5 = external[idx]['external_forces'][:,1,:] # neuron[idx]['us'][:, :]
    # var6 = external[idx]['external_couples'][:,:]
    min_var1 = np.amin(var1)
    max_var1 = np.amax(var1)
    min_var2 = np.amin(var2)
    max_var2 = np.amax(var2)
    min_var3 = []
    max_var3 = []
    for i in range(var3.shape[1]):
        min_var3.append(np.amin(var3[:, i, :]))
        max_var3.append(np.amax(var3[:, i, :]))
    min_var4 = np.amin(var4)
    max_var4 = np.amax(var4)
    min_var5 = np.amin(var5)
    max_var5 = np.amax(var5)
    # min_var6 = np.amin(var6)
    # max_var6 = np.amax(var6)
    min_kappa = np.amin(kappa)
    max_kappa = np.amax(kappa)
    fig = plt.figure(figsize=(30*0.6, 9*0.6))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(3,3,3)#(3, 2, 2)
    ax2 = fig.add_subplot(3,3,6)#(3, 2, 4)
    ax3 = fig.add_subplot(3,3,8)#(3, 2, 6)
    ax21 = fig.add_subplot(3,3,2)
    ax22 = fig.add_subplot(3,3,5)
    ax23 = fig.add_subplot(3,3,9)
    # # ax4 = fig.add_subplot(2, 2, 4)
    # # gs4 = gridspec.GridSpec(3, 3, width_ratios=[0.3, 1.0, 4], height_ratios=[0,2,5])
    # gs4 = gridspec.GridSpec(3, 3, width_ratios=[0.25, 1.2, 4], height_ratios=[3,1,0])
    # ax4 = fig.add_subplot(gs4[4])
    if save_flag:
        factor1 = 2 # 5
        if choice==0:
            name = 'velocity_profiles' # file_name
    else:
        factor1 = int(5000 / save_step_skip) # 50 # 
        name = 'trash'
    fps = 10
    video_name = 'Videos/' + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        start = len(position)-1 # 0
        for k in range(start, len(position)):
            var1 = arm[k]['omega']
            var2 = kappa * _aver(vel_per)
            var3 = muscle[k]['u'][:, :, :]
            var4 = vel_tan # neuron[k]['dist'][:,:]
            var5 = vel_per # neuron[k]['angle'][:,:]
            kappa = arm[k]['kappa'][:,:]
            pos_s_bar = []
            pos_target = []
            for jj in range(int((len(t)-1) / factor1)+1): # +1
                i = jj * factor1
                time = i / (len(t)-1) * final_time
                u = muscle[0]['u'][i, :, :]
                idx = int(neuron[k]['s_bar'][i]) # closest point to the target # np.argmin(var4[i, :]) # 
                pos_s_bar.append(np.array([position[k][i,0,idx], position[k][i,1,idx]]))
                pos_target.append(target[i,:])
                gap = 5
                ax0.cla()
                if flag_obs:
                    alpha0 = 0.8
                    name_obstacle = locals()
                    for o in range(N_obs):
                        name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
                        ax0.add_artist(name_obstacle['obstacle'+str(o)])
                ax0.scatter(position[k][i,0,:],position[k][i,1,:], s=r, marker='o', alpha=1, zorder=2)
                ax0.scatter(position[k][i,0,idx],position[k][i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
                ax0.scatter(np.array(pos_s_bar)[:,0], np.array(pos_s_bar)[:,1], s=5, marker='o', color='C1', alpha=1,zorder=4, label='s_bar')
                ax0.scatter(np.array(pos_target)[:,0], np.array(pos_target)[:,1], s=5, marker='*', color='C2' ,zorder=5, label='target')
                # ax0.plot([np.array(pos_s_bar)[::gap,0], np.array(pos_target)[::gap,0]], [np.array(pos_s_bar)[::gap,1], np.array(pos_target)[::gap,1]], color='k')
                # ax0.quiver(position[k][i,0,idx],position[k][i,1,idx],velocity[i,0,idx],velocity[i,1,idx], angles='xy', scale=1)
                ax0.text(L*0.1, max_var*1.05, 't: %.3f s'%(time), fontsize=12)
                # ax0.scatter(target[i,0], target[i,1], s=100, marker='*', label='target point',zorder=1)
                # # angle = np.linspace(0, 2*np.pi, 100)
                # # dist = norm(target[0,:])
                # # ax0.plot(target[0,0]+dist*np.cos(angle), target[0,1]+dist*np.sin(angle), ls='--', color='black')
                ax1.cla()  # plt.clf()
                ax1.plot(s_mean, var1[i,:])
                # ax1.scatter(s[idx], var1[i,idx-1], s=100, marker='*', color='red')
                ax1.set_xticks([])
                ax1.set_ylabel('$\\omega$', fontsize=12)
                # ax1.set_ylabel('$||v||$', fontsize=12)
                ax2.cla()
                ax2.plot(s[1:-1], var2[i,:])
                ax2.set_xticks([])
                ax2.set_ylabel('$\\kappa v_2$', fontsize=12)
                ax3.cla()
                for ii in range(var3.shape[1]):
                    ax3.plot(s, var3[i, ii, :])
                ax3.vlines(s[idx], 0, np.amax(u), ls='--', lw=1) # s[idx+1]
                ax3.set_ylabel('$u_i$', fontsize=12)
                ax3.set_xlabel('$s$',fontsize=12)
                ax21.cla()  # plt.clf()
                ax21.plot(s_mean, var4[i,:])
                ax21.set_ylabel('$v_1$', fontsize=12)
                # ax21.set_ylabel('distance $\\rho$', fontsize=12)
                ax22.cla()
                ax22.plot(s_mean, var5[i,:])
                ax22.set_ylabel('$v_2$', fontsize=12)
                # ax22.set_ylabel('$\\sin(\\alpha)$', fontsize=12)
                ax23.cla()
                # ax23.plot(s, var6[i,:].T)
                ax23.plot(s[1:-1], kappa[i,:])
                # ax23.scatter(s[idx], kappa[i,idx-2], s=100, marker='*', color='red')
                ax23.set_ylabel('$\\kappa$', fontsize=12)
                ax23.set_xlabel('$s$',fontsize=12)
                ax1.vlines(s[idx], min_var1*1.1, max_var1*1.1, ls='--', lw=1)
                ax2.vlines(s[idx], min_var2*1., max_var2*1., ls='--', lw=1)
                ax21.vlines(s[idx], min_var4-0.1, max_var4+0.1, ls='--', lw=1)
                ax22.vlines(s[idx], min_var5-0.1, max_var5+0.1, ls='--', lw=1)
                ax23.vlines(s[idx], min_kappa, max_kappa*1.2, ls='--', lw=1)
                ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
                ax1.axis([-0.01, L+0.01, min_var1*1.1, max_var1*1.1])
                ax2.axis([-0.01, L+0.01, min_var2*1, max_var2*1])
                ax3.axis([-0.01, L+0.01, min(min_var3)*1-max(max_var3)*0.1, max(max_var3)*1.1])
                ax21.axis([-0.01, L+0.01, min_var4-0.1, max_var4+0.1])
                ax22.axis([-0.01, L+0.01, min_var5-0.1, max_var5+0.1])
                ax23.axis([-0.01, L+0.01, min_kappa*1, max_kappa*1.2])
                # ax0.axis([-0.01, L+0.01, -L, L+0.01])
                # ax1.axis([-0.01, L+0.01, 0., 2.0])
                # ax2.axis([-0.01, L+0.01, 0, 2])
                # ax21.axis([-0.01, L+0.01, -0.01, max_var4*1])
                # ax22.axis([-0.01, L+0.01, -1.0, 1.0])
                # ax23.axis([-0.01, L+0.01, min_var6*1, max_var6*1])
                # ax4.axis([-0.01, L+0.01, min_kappa*1, max_kappa*1.2])
                # ax4.axis([-0.01, L+0.01, 0, 120])
                # ax4.axis([-0.01, L+0.01, min(min_kappa, np.amin(kappa0)), max(max_kappa, np.amax(kappa0))])
                # break
                if not save_flag:
                    plt.pause(0.001)
                else:
                    writer.grab_frame()
            # break
            if not isninf(position[k]):
                break


### single arm moving with closest point trajectory and lines connected to target
elif video == 3:
    max_var = max(L*1.1, np.amax(target[:,0])*1.1) # max(L*1.1, target[0]*1.1) # 1.5
    min_var = min(-L/2, np.amin(target[:,0])*1.1) # /3 # /4
    idx = -1
    min_var_x = min(np.amin(position[idx][0,0,:])*1.1, np.amin(position[idx][-1,0,:])*1.1, min_var*1.01)
    max_var_x = max(np.amax(position[idx][-1,0,:])*1.1, np.amax(position[idx][0,0,:])*1.1, max_var*1.01)
    min_var_y = min(np.amin(position[idx][-1,1,:])*1.1, min_var*1.01)
    max_var_y = max_var*1.01
    dist = neuron[idx]['dist'][:,:]
    fig = plt.figure(figsize=(16, 12))
    ax0 = fig.add_subplot(1, 1, 1)
    if save_flag:
        factor1 = 1000 # min(int(1000 / save_step_skip), 1) # 5
        if choice==0:
            name = file_name # + '_bend_vel'
    else:
        factor1 = int(2000 / save_step_skip)
        name = 'trash'
    fps = 30 # 5 # 10
    video_name = 'Videos/' + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        start = 0 # len(position)-1 # 
        for k in range(start, len(position)):
            pos_s_bar = []
            pos_target = []
            gap = 6
            r *= 10
            for jj in range(int((len(t)-1) / factor1)+1): # +1
                i = jj * factor1
                time = i / (len(t)-1) * final_time
                # idx = np.argmin(dist[i, :])
                idx = int(neuron[k]['s_bar'][i]) # np.argmin(var4[i, :]) # closest point to the target
                pos_s_bar.append(np.array([position[k][i,0,idx], position[k][i,1,idx]]))
                pos_target.append(target[i,:])
                ax0.cla()
                if flag_obs:
                    alpha0 = 0.8
                    name_obstacle = locals()
                    for o in range(N_obs):
                        name_obstacle['obstacle'+str(o)] = plt.Circle((pos_obs[o,0], pos_obs[o,1]), r_obs[o], color='grey', alpha=alpha0)
                        ax0.add_artist(name_obstacle['obstacle'+str(o)])
                # ax0.text(L*0.1, L*0.5, 't: %.3f s'%(time))
                ax0.scatter(position[k][i,0,:],position[k][i,1,:], s=r, marker='o', color='C4', alpha=1,zorder=4)
                ax0.scatter(position[k][i,0,idx],position[k][i,1,idx], s=r[idx], marker='o', color='C0', alpha=1,zorder=5)
                ax0.scatter(target[i,0], target[i,1], s=300, marker='*', color='C3', label='target',zorder=3)
                ## trajectories of s_bar point and target point
                ax0.scatter(np.array(pos_target)[:,0], np.array(pos_target)[:,1], s=40, marker='*', color='C1' ,zorder=1)
                ax0.scatter(np.array(pos_s_bar)[:,0], np.array(pos_s_bar)[:,1], s=40, marker='o', color='C2', alpha=1,zorder=2) # , label='s_bar'
                ax0.plot([np.array(pos_s_bar)[::gap,0], np.array(pos_target)[::gap,0]], [np.array(pos_s_bar)[::gap,1], np.array(pos_target)[::gap,1]], color='k')
                ax0.text(L*0.1, max_var*1.05, 't: %.3f s'%(time), fontsize=12)
                # ax0.legend(bbox_to_anchor=(1, 1), loc=4, ncol=2, borderaxespad=0)
                # plt.legend(frameon=False)
                # ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
                ax0.axis([-0.01, 0.2, -0.02, 0.15])
                plt.axis('off')
                if not save_flag:
                    plt.pause(0.001)
                else:
                    writer.grab_frame()
            # break
            if not isninf(position[k]):
                break


### velocity profile of experiment and simulation
elif video == 4:
    fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(16, 9))
    if save_flag:
        factor1 = 1 # min(int(1000 / save_step_skip), 1) # 5
        if choice==0:
            name = 'bend_vel_comparison'
    else:
        factor1 = 1
        name = 'trash'
    fps = 50 # 10
    video_name = 'Videos/' + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        data_vel = np.load('Data/bend_vel_exp1.npy', allow_pickle='TRUE').item()
        t_exp = data_vel['t1']
        bend_vel_exp = data_vel['bend_vel']
        bend_vel_exp_filtered = data_vel['bend_vel_filtered']
        bend_velocity, bend_pos = Bell_Vel() # Bend_Vel()
        bend_velocity = bend_velocity[:-5]
        t_t = t[:len(bend_velocity)]
        bend_vel = abs(bend_velocity)
        vel_filtered = low_pass_filter(bend_vel)
        count = 0
        for jj in range(len(t_t)):
            i = jj * factor1
            if i >= (count)/44*145:
                count += 1
            ax.cla()
            # ax.plot(t_t, bend_vel, label='simulation')
            # ax.plot(t_exp, bend_vel_exp, label='experiment')
            ax.plot(t_t, vel_filtered, lw=4, color='C0', alpha=0.3)
            ax.plot(t_exp, bend_vel_exp_filtered, lw=4, color='C1', alpha=0.3)
            ax.plot(t_t[:i+1], vel_filtered[:i+1], lw=4, color='C0', label='simulation')
            ax.plot(t_exp[:count], bend_vel_exp_filtered[:count], lw=4, color='C1', label='experiment')
            # ax.legend()
            # ax.set_xlabel('t (s)')
            # ax.set_ylabel('bend velocity (m/s)')
            ax.legend(bbox_to_anchor=(1, 1), loc=4, ncol=2, borderaxespad=0)
            ax.axis([0., 1.5, 0., 0.16])
            ax.set_xticks([0,0.5,1.0,1.5])
            ax.set_yticks([0,0.08,0.16])
            ax.set_rasterized(True)
            if not save_flag:
                plt.pause(0.001)
            else:
                writer.grab_frame()

plt.show()
