import numpy as np 
from scipy.io import loadmat
import skimage.data
import matplotlib.pyplot as plt 


a = skimage.data.imread("img/ecg_fish0.jpg")

m = np.min(a, 2)
plt.imshow(m)
plt.show()
