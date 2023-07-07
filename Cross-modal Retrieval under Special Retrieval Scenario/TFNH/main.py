import tensorflow as tf
from TFNH import TFNH
import numpy as np

with tf.Session() as sess:
    model = TFNH(sess)
    for i in np.arange(10):
        print('training ' + str(i+1))
        model.train_model()