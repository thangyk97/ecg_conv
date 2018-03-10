from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np 
import skimage.data
import os
import time
"""
Cut ecg signal to images of beat
@arg: r index, ecg
@output: array of image (numpy)
"""
def cut_beat(index_R, data_ecg):
    list_ = [] # store images of 1 signal
    # range(len(index_R))
    for i in range(len(index_R)):
        try:
            # Get segment with r peak
            temp = data_ecg[index_R[i+1] - 180: index_R[i+1] + 180]
        except IndexError:
            continue

        # Create image and save
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.plot(range(temp.shape[0]), temp, c='grey', linewidth=6)
        fig.savefig('img/ecg_fish' + str(i) + '.jpg',
                    dpi=32
        )
        plt.close()
        # Load image to save in npy file
        global a
        a = skimage.data.imread('img/ecg_fish' + str(i) + '.jpg')
        m = np.min(a, 2)
        list_.append(m)

    if not list_: # Check if list_ is empty
        return False, 0

    out = np.array(list_)
    out = np.reshape(out, [-1, a.shape[0], a.shape[1]]) # Reshape to append
    print (out.shape)
    return True, out

# Load ecg signal and index of R peaks
path_dir = '/home/thangkt/git/fish/PROCESSED ECG database/processedSA'
list_dirs = [d for d in os.listdir(path_dir)
                if d.endswith(".txt")]
list_dirs = np.sort(list_dirs)
data_index_R = loadmat('r_index/processedSA.mat')
# end load
start = time.time()

# Loop with all file ecg
for i in range(0, len(list_dirs)):
# for i in range(5, 6):
    data_ecg = np.loadtxt(os.path.join(path_dir, list_dirs[i])) # ECG signal
    index_R = data_index_R['r_index'][0][i][0]                  # index of r peaks
    if len(index_R) == 0:       # Check if signal has no r peaks
        print ("Can't process signal hasn't r peak")
    else:
        print (index_R.shape)
        # Segmentes with r peaks - numpy array [-1, img.shape[0], img.shape[1]]
        check, a = cut_beat(index_R, data_ecg) 
        if check:
            if i == 0:                # first loop
                list_ = a
            else:
                list_ = np.append(list_, a, 0)
    
    print ("file " + str(i) + list_dirs[i])

print (list_.shape)
np.save("train/sa.npy", list_) # Save images as npy file
print ("Successful!")
end = time.time()
print ("Time: {}".format(end - start))