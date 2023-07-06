import numpy as np
import scipy.io as sio
import warnings
import dataset.nuswide as dataset
import cdq as model
import sys

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

model_weight = sys.argv[1]
subspace_n = int(sys.argv[2])
output_dimension = int(sys.argv[3])
update_b_n = int(sys.argv[4])
gpu = sys.argv[5]

config = {
    'device': '/gpu:' + gpu,
    'centers_device': '/gpu:' + gpu,
    'training_epoch': 30,
    'max_iter': 5000,
    'max_iter_update_b': update_b_n,
    'max_iter_update_Cb': 1,
    'cq_lambda': 0.1,
    'alpha': 1.0,
    'batch_size': 100,
    'code_batch_size': 100,
    'moving_average_decay': 0.9999,      # The decay to use for the moving average. 
    'num_epochs_per_decay': 5,          # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.1,   # Learning rate decay factor.
    'initial_learning_rate_img': 0.001,       # Initial learning rate img.
    'initial_learning_rate_txt': 0.0001,       # Initial learning rate txt.

    # 2^8 * 4 = 32bit
    'n_subspace': subspace_n,
    'n_subcenter': 256,
    'output_dim': output_dimension,

    'R': 50,
    'model_weights': model_weight,
    'save': False,
    'shuffle': True,

    'train': False,

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

query_img, query_txt, database_img, database_txt = dataset.import_validation(config)

model_dq = model.cdq(config)

model.validation(database_img, database_txt, query_img, query_txt, model_dq, config)
