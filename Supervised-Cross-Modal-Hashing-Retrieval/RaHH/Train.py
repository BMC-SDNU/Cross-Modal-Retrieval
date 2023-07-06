__author__ = 'Bo Liu'
from numpy import *
from loss_func import *
from update import *

def train(img_fea, tag_fea, H_img, H_tag, S, W, R_pq, R_p, R_q, OutofSample, up_mp, up_mq, parameter):
    #print 'Train func begin'

    alpha = parameter['alpha']
    beta = parameter['beta'] #heterogeneous
    gamma1 = parameter['gamma1']#1e-1 #regularization 1
    gamma2 = parameter['gamma2']#1e-1 #regularization 2
    gamma3 = parameter['gamma3']#1e-1 #regualariation 3
    lambda_w = parameter['lambda_w']
    lambda_h =  parameter['lambda_h']
    lambda_reg = parameter['lambda_reg']
    converge_threshold = 1e-8 #/ sqrt(img_fea.shape[1] * tag_fea.shape[1])
    #print 'converge_threshol:', converge_threshold

    #print 'begin to calculate loss func'
    new_loss = loss_func(img_fea, tag_fea, H_img, H_tag, R_pq, R_p, R_q, W, S, alpha, beta, gamma1, gamma2, gamma3)
    old_loss = new_loss + 20  # just for start

    fea = [img_fea, tag_fea]
    H = [H_img, H_tag]

    W = W.transpose()
    R_pq = R_pq.transpose()

    #print '---------Training---------------'

    iteration = 0

    #while (abs(old_loss - new_loss) > converge_threshold) and (iteration < 70):
    while (old_loss - new_loss > 1e-4):
        iteration += 1
        #print '-------------------------------'
        #print iteration, 'times iteration'

        old_loss = new_loss
        #print old_loss
        #update the hash code
        #and update the statistics S
        for p in range(2):
            q = 1 - p
            W = W.transpose()
            R_pq = R_pq.transpose()

            [H, S] = update_h(fea, H, W, S, R_pq, R_p.transpose(), R_q.transpose(), p, alpha, beta, gamma1, gamma2, gamma3, lambda_h, OutofSample, up_mp, up_mq)
            if not OutofSample:
                W = update_w(H, R_pq, W, p, lambda_w, lambda_reg)

        new_loss = loss_func(img_fea, tag_fea, H[0], H[1], R_pq.transpose(), R_p.transpose(), R_q.transpose(), W.transpose(), S, alpha, beta, gamma1, gamma2, gamma3)
        #print 'new loss', new_loss

    H_img = sign(H[0])
    H_tag = sign(H[1])

    return [H_img, H_tag, W.transpose(), S]
