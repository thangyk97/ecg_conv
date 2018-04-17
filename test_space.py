import numpy as np 
import matplotlib.pyplot as plt 
import time
import os
import skimage.data
from scipy.io import loadmat

##### Set up ######
file            = loadmat('r_index/processedST.mat')
ecg_signal_path = '/home/thangkt/git/fish/PROCESSED ECG database/processedST'
save_path       = 'processedST.npy'
Hz              = 1000
segment_len     = 4 * Hz

# Compute
nb_record       = file['output'][0].shape[0]
left_len = right_len = segment_len // 2
print (left_len, right_len)

###############

# Loop per record
for i in range(nb_record):
    name, r_data = file['output'][0][i][0][0][0], file['output'][0][i][0][1][0] # multi zero index since file structure, ignore it !
    ecg = np.loadtxt(os.path.join(ecg_signal_path, name)) # Load fish-ecg data
    plt.plot(list(range(len(ecg))), ecg)
    plt.scatter(r_data, ecg[r_data])
    plt.show()
    break