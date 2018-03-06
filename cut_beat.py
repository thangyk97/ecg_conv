from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np 



# Load ecg signal and index of R peaks
data_ecg = np.loadtxt(
    '/home/thangkt/git/fish/PROCESSED ECG database/processedSA/processedSA_100_1.txt'
)
data_index_R = loadmat('r_index/processedSA.mat')


index_R = data_index_R['r_index'][0][0][0]
# range(len(index_R))
for i in range(1):
    try:
        temp = data_ecg[index_R[i+1] - 18: index_R[i+1] + 18]
    except IndexError:
        break

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.plot(range(temp.shape[0]), temp, c='grey', linewidth=6)

    fig.savefig('img/ecg_fish' + str(i) + '.jpg',
                dpi=32
    )

    plt.close()

print ("Successful!")