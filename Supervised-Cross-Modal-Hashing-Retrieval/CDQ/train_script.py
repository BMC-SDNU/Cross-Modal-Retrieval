import numpy as np
import scipy.io as sio
import warnings
import dataset.nuswide as dataset
import cdq as model
import sys

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

lr_img = float(sys.argv[1])
lr_txt = float(sys.argv[2])
cq_lambda = float(sys.argv[3])
update_b = int(sys.argv[4])
subspace_n = int(sys.argv[5])
output_dimension = int(sys.argv[6])
gpu = sys.argv[7]

config = {
    'device': '/gpu:' + gpu,
    'centers_device': '/gpu:' + gpu,
    'training_epoch': 30,
    'max_iter': 5000,
    'max_iter_update_b': update_b,
    'max_iter_update_Cb': 1,
    'cq_lambda': cq_lambda,
    'alpha': 1.0,
    'batch_size': 64,
    'code_batch_size': 100,
    'moving_average_decay': 0.9999,      # The decay to use for the moving average. 
    'num_epochs_per_decay': 15,          # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.5,   # Learning rate decay factor.
    'initial_learning_rate_img': lr_img,       # Initial learning rate img.
    'initial_learning_rate_txt': lr_txt,       # Initial learning rate txt.

    # 2^8 * 4 = 32bit
    'n_subspace': subspace_n,
    'n_subcenter': 256,
    'output_dim': output_dimension,

    'R': 50,
    'weights': 'models/bvlc_alexnet.npy',
    'save': False,
    'shuffle': True,

    'img_model': 'alexnet',
    'txt_model': 'mlp',

    'n_train': 10500,
    'n_database': 150300,
    'n_query': 2000,

    'txt_dim': 1000,
    'label_dim': 21,
    'img_tr': "data/nuswide/train.txt",
    'txt_tr': "data/nuswide/text_train.txt",
    'img_te': "data/nuswide/test.txt",
    'txt_te': "data/nuswide/text_test.txt",
    'img_db': "data/nuswide/database.txt",
    'txt_db': "data/nuswide/text_database.txt",
    'save_dir': "models/",
}

train_img, train_txt = dataset.import_train(config)

model_dq = model.train(train_img, train_txt, config)
