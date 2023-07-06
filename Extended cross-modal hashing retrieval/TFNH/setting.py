import scipy.io as sio

# WIKI
dataset = 'data/WikiData.mat'

# batch_size
pair = True
pair_batch_size = 217
img_batch_size = 217
txt_batch_size = 217
BATCH_NUM = 10
batch_size = pair_batch_size + img_batch_size + txt_batch_size

# epoch
TOTAL_EPOCH = 200
kp = 0.5

# network structures setting
hid_dim = 1024
hash_dim = 64
dis_dim = 64
dis_out_dim = 2

# hyper-parameters setting
alpha = 1
beta = [0.1, 0.3, 0.3]
mu = 2
yita = [20, 20]
lr_fn = 0.001
lr_dn = 0.001

# load data
data = sio.loadmat(dataset)
I_tr = data['I_tr']
T_tr = data['T_tr']
L_tr = data['L_tr']
I_te = data['I_te']
T_te = data['T_te']
L_te = data['L_te']
del data

# input dimension
img_dim = I_tr.shape[1]
txt_dim = T_tr.shape[1]
lab_dim = L_tr.shape[1]
all_num = L_tr.shape[0]
fusion_dim = img_dim + txt_dim
    



















