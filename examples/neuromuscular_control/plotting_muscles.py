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

def isninf(a):
    return np.all(np.isfinite(a))

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

    folder_name = 'Data/'
    if case == 'LM_reach':
        file_name = 'journal_neural_reaching_LM'
    elif case == 'LM_point':
        file_name = 'journal_neural_pointing_LM'
    elif case == 'TM_reach':
        file_name = 'journal_neural_reaching_TM'

    data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

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
    mu = data['model']['sensor_mag']

    # list_name = [0.04, 0.02, 10*tau, 0., 1., 60, 40, 80, 0] # negative lambda means tapered
    # data_init = {
    #     'pos': np.vstack([position[0][-1,:,:], np.zeros(n_elem+1)]), 
    #     'rest_kappa': np.vstack([-arm[0]['kappa'][-1,:], np.zeros([2, n_elem-1])]),
    #     'rest_V': neuron[0]['V'][-1,:,:],
    #     'model': data['model'],
    # }
    # np.save('Data/initial_data_'+str(list_name[4])+'-'+str(list_name[5])+'-'+str(list_name[7])+'.npy', data_init)
    # quit()

    actual_list = [
        data['model']['neuron']['tau'], data['model']['neuron']['lmd'], 
        data['model']['neuron']['tau_adapt'], data['model']['neuron']['inhibition'], 
        data['model']['neuron']['adaptation'], 
        data['model']['neuron']['V0'][0], data['model']['neuron']['V0'][1], 
        data['model']['neuron']['V0'][2], data['model']['neuron']['V0'][3]]
    print('actual_list', actual_list)

    print('total_steps: ', len(t), ', final_time=', final_time)
    print('target=', target[0,:])

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
        i = 0
        ax0.scatter(position[-1][i,0,:],position[-1][i,1,:], s=r, marker='o', alpha=1,zorder=2)
        i = -1
        ax0.scatter(position[-1][i,0,:],position[-1][i,1,:], s=r, marker='o', alpha=1,zorder=2)
        if flag_target:
            ax0.scatter(target[i,0], target[i,1], s=100, marker='*', label='target point',zorder=1)
        ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
        ax0.axis('off')

    plot = 1
    flag_savefig = 0
    if plot:
        plot_arm()

    video = 2 # 2 # 3
    save_flag = 0

    ## V and I from cable equation
    if video == 2:
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
            factor1 = 1
            name = file_name
        else:
            factor1 = 5
            name = 'trash'
        slow_factor = 2
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
                        ax3.plot(s, var1[i, ii, :]/mu, ls='--', color='k')
                    ax4.plot(s[1:-1], kappa[i,:])
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

    plt.show()


if __name__ == "__main__":
	main()