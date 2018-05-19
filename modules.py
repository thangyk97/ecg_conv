import matplotlib.pyplot as plt 
import numpy as np
import time

"""
Contain function process for files in ECG 
"""
class function(object):
    def __init__(self, *args):
        super(function, self).__init__(*args)
        
    """
    Display images has same label
    @param: 3darray images, 1darray labels, int label

    """
    def display_images_has_label(images, labels, label, title = ""):
        limit = 24
        index = np.where(labels == label)[0]
        np.random.shuffle(index)
        index = index[:limit]
        images_part = images[index]

        plt.figure(figsize=(15, 5))
        plt.suptitle(title)
        i = 1
        for image in images_part:
            plt.subplot(3, 8, i)
            plt.axis('off')

            i += 1
            plt.imshow(image)
        plt.show()

    

    