import tensorflow as tf
from train import CrossModal, ModelParams
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(_):
    graph = tf.Graph()
    model_params = ModelParams()
    model_params.update()

    with graph.as_default():
        model = CrossModal(model_params)
    with tf.Session(graph=graph,config=config) as sess:
        model.train(sess)


if __name__ == '__main__':
    tf.app.run()
