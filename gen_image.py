import numpy as np 
import matplotlib.pyplot as plt 
import time
import os
import skimage.data
from scipy.io import loadmat

##### Set up ######
file            = loadmat('r_index/processedHighP.mat')
ecg_signal_path = '/home/thangkt/git/fish/PROCESSED ECG database/processedHighP'
save_path       = 'processedST.npy'
Hz              = 1000
segment_len     = 1 * Hz

# Compute
nb_record       = file['output'][0].shape[0]
left_len = right_len = segment_len // 2

###############

list_ = [] # store images of 1 signal
# Loop per record
for i in range(nb_record):
    name, r_data = file['output'][0][i][0][0][0], file['output'][0][i][0][1][0] # multi zero index since file structure, ignore it !
    ecg = np.loadtxt(os.path.join(ecg_signal_path, name)) # Load fish-ecg data

    if len(ecg) < segment_len: continue

    flag = False # Check cut segment
    # Loop per r_peak
    for k in range(len(r_data)):
        
        if r_data[k] < left_len: # Only firt R peak which has index < left lenght 
            if k != 0: continue
            segment = ecg[0:segment_len]
        elif r_data[k] > len(ecg) - right_len - 2: # Only firt R peak which has index > right lenght
            if flag: break
            segment = ecg[-segment_len:]
        else:
            segment = ecg[r_data[k] - left_len: r_data[k] + right_len]
            
        # Create image and save
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.plot(range(segment.shape[0]), segment, c='grey', linewidth=4)
        fig.savefig('img/ecg_fish' + str(k) + '.jpg',
                    dpi=128)

        plt.close()
        # Load image to save in npy file
        global a
        a = skimage.data.imread('img/ecg_fish' + str(i) + '.jpg')
        m = np.min(a, 2)
        list_.append(m)
    print (len(list_))
    if i == 0: break
    

    