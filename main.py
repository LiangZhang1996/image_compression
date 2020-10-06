import tensorflow as tf 
imoprt tensorflow.keras as keras
from mymodel import *
import numpy as np 
import matplotlib.pyplot as plt 


def read_image(path, n):
    # make the images become datasets that can be tarined 
    # input the images' path and number
    data = []
    files = [path+str(i)+'.png' for i in range(n)]
    for f in files:
        image = plt.imread(f)
        data.append(image[:,:,0:3])
    return np.array(data)


# img infer the origin image and new_img infer the decompressed imgae
def MSE(img, new_img):
    return tf.reduce_mean(tf.square(img, new_img))

def PNSR(img, new_img):
    mse = MSE(img, new_img)
    return 10*np.log10(1/mse)


if __name__ == '__main__':

