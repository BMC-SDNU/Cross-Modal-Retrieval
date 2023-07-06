__author__ = 'Bo Liu'
from numpy import *
from loss_func import *

def update_h(fea, H, W, S, R_pq, Rp, Rq, p, alpha, beta, gamma1, gamma2, gamma3, lambda_h, OutofSample, up_mp, up_mq):
    #Does not consider the homogeneous similarity
    #Thus the part to upate the homogeneous is ignored.

    #used for two domain situation, only, currently
    R = [Rp, Rq]
    up = [up_mp, up_mq]
    q = 1 - p

    rp = shape(H[p])[0]
    mp = shape(H[p])[1]

    rq = shape(H[q])[0]
    mq = shape(H[q])[1]

    #The derivative
    Gradient = zeros([rp, mp])

    gd_1 = 4 * ((H[p] * H[p] - eye(rp, mp)) * H[p])
    gd_2 = multiply(2, S[p][1]) #rp \times 1
    gd_2 = tile(gd_2, (1, mp))
    gd_3 = multiply(4, dot(S[p][2], H[p]))

    #Gradient = Gradient + gamma1 * gd_1 + gamma2 * gd_2 + gamma3 * gd_3

    #Homogeneous
    Ap = dot(fea[p].transpose(), fea[p])
    ap_1 = dot(Ap, ones((mp, 1)))
    ap_1.shape = [mp, ]
    ap_diag = diag(ap_1)

    gd_homo = dot(H[p], ap_diag)
    gd_homo = gd_homo - dot(dot(H[p], fea[p].transpose()), fea[p])# - alpha * dot(H[p], R[p])
    Gradient += alpha * gd_homo

    Hq_map = dot(W.transpose(), H[p]) #hash code of q acquired from mapping Hp
    Hp_map = dot(W, H[q]) #hash code of p acquired from mapping Hq

    if OutofSample and up[p] != 0:
        update_range = range(up[p], mp)
    else:
        if OutofSample and up[p] == 0:
            update_range = []
        else:
            update_range = range(mp)

    for k in range(rp):
        #print 'Update H:', k

        for i in update_range:
            gd = 0

            R_pqij = tile(R_pq[i, :], (rq, 1))
            H_qgj = H[q]
            W_pqkg = tile(W[k, :].transpose(), (mq, 1)).transpose()
            Wg_Hip = tile(Hq_map[:, i], (mq, 1)).transpose()
            gd += sum((-R_pqij * H_qgj * W_pqkg) / (1 + exp(R_pqij * H_qgj * Wg_Hip)))
            gd += sum((-R_pq[i, :] * Hp_map[k, :]) / (1 + exp(R_pq[i, :] * Hp_map[k, :] * H[p][k, i])))

            Gradient[k, i] += gamma1 * gd_1[k, i] + gamma2 * gd_2[k, i] + gamma3 * gd_3[k, i]
            Gradient[k, i] += beta * gd

        H[p][k, :] = H[p][k, :] - lambda_h * Gradient[k, :] / (mq * mp * (rp + rq))# * (alpha + beta + gamma1 + gamma2 + gamma3))

        if not OutofSample:
            S[p] = update_S(fea[p], H[p])

    return [H, S]


def update_w(H, R_pq, W, p, lambda_reg, lambda_w):
    q = 1 - p

    mp = H[p].shape[1]
    mq = H[q].shape[1]

    rp = H[p].shape[0]
    rq = H[q].shape[0]

    Hq_maped = dot(W.transpose(), H[p])

    for k in range(rq):

        #mp \times 1
        Rpq_Hkj = R_pq * tile(H[q][k, :], (mp, 1))

        gd_i = (-Rpq_Hkj) / (1 + exp(Rpq_Hkj * tile(Hq_maped[k, :], (mq, 1)).transpose()))
        gd_i = sum(gd_i, 1)
        gd = dot(H[p], gd_i.transpose())
        gd += 2 * lambda_reg * W[:, k]

        W[:, k] = W[:, k] - lambda_w * gd / (mp * mq * (rp + rq)) #* (alpha + beta + gamma1 + gamma2 + gamma3))

    return W


def update_S(fea, hash):

    S_0 = dot(hash, transpose(fea))
    S_1 = dot(hash, ones([shape(fea)[1], 1]))
    S_2 = subtract(dot(hash, transpose(hash)), multiply(shape(fea)[1], eye(shape(hash)[0])))

    return [S_0, S_1, S_2]
