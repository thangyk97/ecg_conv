import numpy as np 
import matplotlib.pyplot as plt 
import time
from modules import function

start = time.time()


images = np.load("train/highp.npy")
labels = np.zeros(shape=[images.shape[0]])
fn = function()
fn.display_images_has_label(images=images, labels=labels, label=0)


print (images.shape)


end = time.time()
print ("time: {}".format(end - start))