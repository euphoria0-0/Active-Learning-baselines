# status initialize

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #// 초기화할 GPU number



# out of memory


import tensorflow as tf

with tf.Graph().as_default():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)