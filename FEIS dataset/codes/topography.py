'''
Objective: Static and animation 2D topographic map for brain mapping using power calculated by Welch method
           for Emotiv EPOC 14 channel headset
Input: csv file (FEIS dataset folder traversing is done)
Output: Static 2D topographic map for 21 participants on single phoneme or 
        Animation 2D topographic map for 1 participant
@author: Vinay
Adapted from: https://github.com/ijmax/EEG-processing-python
'''

import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import signal
import scipy.interpolate
from matplotlib import patches
import matplotlib.pyplot as plt

def get_psds(ch_data, fs = 256, f_range= [0.5, 30]):
    '''
    Calculate signal power using Welch method.
    Input: data- mxn matrix (m: number of channels, n: samples of signals)
           fs- Sampling frequency (default 256Hz)
           f_range- Frequency range (default 0.5Hz to 30Hz)
    Output: Power values and PSD values
    '''
    powers = []
    psds = list()
    for sig in ch_data:
        freq, psd = signal.welch(sig, fs)
        idx = np.logical_and(freq >= f_range[0], freq <= f_range[1])
        powers = np.append(powers, sum(psd[idx]))
        psds.append(psd[idx])
    return powers, psds

def plot_topomap(data, ax, fig, draw_cbar=False):
    '''
    Plot topographic plot of EEG data. This is specially design for Emotiv 14 electrode data. This can be change for any other arrangement by changing 
    ch_pos (channel position array) 
    Input: data- 1D array 14 data values
        ax- Matplotlib subplot object to be plotted every thing
        fig- Matplot lib figure object to draw colormap
        draw_cbar- Visualize color bar in the plot
    '''
    N = 300            
    xy_center = [2,2]  
    radius = 2 
    
    # AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    ch_pos = [[1,4],[0.1,3], [1.5,3.5], 
              [0.5,2.5], [-0.1,2], [0.4,0.4], 
              [1.5,0], [2.5,0], [3.6,0.4], [4.1,2], 
              [3.5,2.5], [2.5,3.5], [3.9,3], [3,4]]
    
    x,y = [],[]
    for i in ch_pos:
        x.append(i[0])
        y.append(i[1])
    
    xi = np.linspace(-2, 6, N)
    yi = np.linspace(-2, 6, N)
    zi = scipy.interpolate.griddata((x, y), data, (xi[None,:], yi[:,None]), method='cubic')
    
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"
        
    dist = ax.contourf(xi, yi, zi, 60, cmap = plt.get_cmap('coolwarm'), zorder = 1)
    ax.contour(xi, yi, zi, 15, linewidths = 0.5,colors = "grey", zorder = 2)
    
    if draw_cbar:
        cbar = fig.colorbar(dist, ax=ax, format='%.1e')
        cbar.ax.tick_params(labelsize=8)
    
    ax.scatter(x, y, marker = 'o', c = 'k', s = 10, zorder = 3)
    circle = patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none", zorder=4)
    ax.add_patch(circle)
    
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)

    ax.set_xticks([])
    ax.set_yticks([])
    
    circle = patches.Ellipse(xy = [0,2], width = 0.4, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = patches.Ellipse(xy = [4,2], width = 0.4, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
        
    xy = [[1.6,3.6], [2,4.3],[2.4,3.6]]
    polygon = patches.Polygon(xy = xy, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon) 

    #ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.2, 4.2)
    return ax


# Static visualization for 21 participant
rootdir = 'C:/Users/vinay/Downloads/FEIS_v1_1/experiments/'
fig = plt.figure(figsize=(8,10))
fig.subplots_adjust(hspace=0.5)
fig.suptitle("Topograph for phoneme 'p' in articulator phase", fontsize=15, y=0.95)
i = 1
for subdir, dirs, files in os.walk(rootdir):
    splitted = Path(subdir).parts
    if splitted[-1] == 'p' and splitted[-2] == 'articulators_eeg':
        for file in files: # reading only 1st file many files
            path = subdir + '/' + file
            data = pd.read_csv(path)
            ch_data = np.transpose(data.to_numpy())
            pwrs, _ = get_psds(ch_data)
            ax = plt.subplot(7, 3, i)
            plot_topomap(pwrs, ax, fig)
            ax.set_title("participant " + str(i))
            i += 1
            break
plt.show()
fig.savefig("topograph_p_articulator.png", dpi=300)

'''
# Animation for only 1 participant
path = 'C:/Users/vinay/Downloads/FEIS_v1_1/experiments/01/thinking_eeg/f/thinking_f_trial_4.csv'
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))
data = pd.read_csv(path)
ch_data = np.transpose(data.to_numpy())
chunk_data = np.array_split(ch_data, 4, axis=1)
for chunk in chunk_data:   
    pwrs, _ = get_psds(chunk)
    ax.clear()     
    plot_topomap(pwrs, ax, fig, draw_cbar=False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)
'''