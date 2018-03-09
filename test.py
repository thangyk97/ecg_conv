import numpy as np 
from scipy.io import loadmat
import skimage.data
import matplotlib.pyplot as plt 
import time

start = time.time()

a = np.load('train/st.npy')
m = a[450]
plt.imshow(m)
end = time.time()
print ("time: {}".format(end - start))
plt.show()


