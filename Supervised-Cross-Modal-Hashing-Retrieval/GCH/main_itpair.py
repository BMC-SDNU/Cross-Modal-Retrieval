import os
from setting import *
from GH_itpair import GH
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = select_gpu

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))

with tf.Session(config=gpuconfig) as sess:
    model = GH(sess)
    model.Train() if phase == 'train' else model.test()
