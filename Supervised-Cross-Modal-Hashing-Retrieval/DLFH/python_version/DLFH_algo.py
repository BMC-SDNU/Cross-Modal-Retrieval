#!/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: DLFH_algo.py
# @Author: Qing-Yuan Jiang
# @Mail: qyjiang24 AT gmail.com
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np

from utils.args import args


def __update_columnU__(U, V, sim, sample_idx, num_samples):
    m = num_samples
    lamda = args.lamda
    for k in range(args.bit):
        TX = lamda * U.dot(V[sample_idx, :].T) / args.bit
        AX = 1. / (1 + np.exp(-TX))
        Vjk = V[sample_idx, k].transpose()
        p = lamda * ((sim - AX) * Vjk).dot(np.ones((m, 1))) / args.bit + \
            (m * lamda ** 2 * U[:, k] / (4 * args.bit ** 2)).reshape(-1, 1)
        U_opt = np.sign(p)
        U[:, k] = U_opt.squeeze()
    return U


def __update_columnV__(V, U, sim, sample_idx, num_samples):
    m = num_samples
    lamda = args.lamda
    for k in range(args.bit):
        TX = lamda * U[sample_idx, :].dot(V.T) / args.bit
        AX = 1. / (1 + np.exp(-TX))
        Ujk = U[sample_idx, k].transpose()
        p = lamda * ((sim.T - AX.T) * Ujk).dot(np.ones((m, 1))) / args.bit + \
            (m * lamda ** 2 * V[:, k] / (4 * args.bit ** 2)).reshape(-1, 1)

        V_opt = np.sign(p)
        V[:, k] = V_opt.squeeze()
    return V


def dlfh_algo(train_labels):
    num_train = train_labels.shape[0]
    num_samples = args.bit

    U = np.sign(np.random.randn(num_train, args.bit))
    V = np.sign(np.random.randn(num_train, args.bit))

    for iter in range(args.max_iter):
        sample_idx = list(np.random.permutation(num_train))[0: num_samples]
        train_label = train_labels[sample_idx]
        sim = (train_labels.dot(train_label.T) > 0).astype(np.float)
        U = __update_columnU__(U, V, sim, sample_idx, num_samples)

        sim = (train_label.dot(train_labels.T) > 0).astype(np.float)
        V = __update_columnV__(V, U, sim, sample_idx, num_samples)
    return U, V
