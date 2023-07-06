import load_data
import numpy as np
from cvh import cvh
from loss_func import *
from Test import *
from OutSample import *
from init import *
from Train import *
import time

#Relation-aware Heterogeneous Hashing(RaHH)
#Puesdo-Code
#Author: Bo Liu
#Date: Oct. 05 2013
#References:
#Ou, M., Cui, P., Wang, F., Wang, J., Zhu, W., & Yang, S. (2013). Comparing apples to oranges: a scalable solution with heterogeneous hashing.
#Paper presented at the Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining.
#Rahh()
#Input: X^p/data, R_p,intra_domain relation, R_pq inter-domain relation. r: the number of bit for each domain
#Output: H^p: hash function, W: map function to map the hash code to another Hamming space.
def subsampling(fea1, fea2, sim, lin, num):

    if lin == 0:
        fea1_ind = random.random_integers(0, fea1.shape[1] - 1, num)
        fea2_ind = random.random_integers(0, fea2.shape[1] - 1, num)
    else:
        if num == 0:
            fea1_ind = random.random_integers(0, fea1.shape[1] - 1, fea1.shape[1] / lin)
            fea2_ind = random.random_integers(0, fea2.shape[1] - 1, fea2.shape[1] / lin)

    fea1 = fea1[:, fea1_ind]
    fea2 = fea2[:, fea2_ind]
    sim = (sim[fea1_ind, :])[:, fea2_ind]

    return fea1, fea2, sim

def RaHH(bit, output):
    #R is the similarity to keep the consistent with origin paper
    parameter = {}
    parameter['alpha'] = 1
    parameter['beta'] = 100#heterogeneous
    parameter['gamma1'] = 10#1e-1 #regularization 1
    parameter['gamma2'] = 1#1e-1 #regularization 2
    parameter['gamma3'] = 3#1e-1 #regualariation 3
    parameter['lambda_reg'] = 1
    #learning rate
    parameter['lambda_h'] = 0.1
    parameter['lambda_w'] = 0.1

    #Tr_sim_path = 'Data/Train/similarity.txt'
    #Tr_img_path = 'Data/Train/images_features.txt'
    #Tr_tag_path = 'Data/Train/tags_features.txt'
    #Tst_img_path = 'Data/Test/images_features.txt'
    #Tst_qa_path = 'Data/Test/QA_features.txt'
    #gd_path = 'Data/Test/groundtruth.txt'
    Tr_img_path = 'Data/Subset/Training/300/images_features.txt'
    Tr_tag_path = 'Data/Subset/Training/300/tags_features.txt'
    Tr_sim_path = 'Data/Subset/Training/300/similarity.txt'
    Tst_img_path = 'Data/Subset/Query/images_features.txt'
    Tst_qa_path = 'Data/Subset/Test/tags_features.txt'
    gd_path = 'Data/Subset/Query/groundtruth.txt'

    [Tr_sim, Tr_img, Tr_tag, Tst_img, Tst_qa, gd] = load_data.analysis(Tr_sim_path, Tr_img_path, Tr_tag_path, Tst_img_path, Tst_qa_path, gd_path)
    #image_fea d_p * m_p
    #tag_fea d_q* m_q
    #similarty : m_p * m_q
    #QA_fea = d_p * m_p
    #GD = #img * #QA
    #Tr_img, Tr_tag, Tr_sim = subsampling(Tr_img, Tr_tag, Tr_sim, 0, 300)


    #print '----------------CVH finish----------------------'
    print time.clock()
    [H_img, H_tag, W, S, R_p, R_q, A_img, A_tag] = initialize(Tr_img, Tr_tag, Tr_sim, bit)

    #print 'begin RaHH train'
    [H_img, H_tag, W, S] = train(Tr_img, Tr_tag, H_img, H_tag, S, W, Tr_sim, R_p, R_q, False, 0, 0, parameter)

    GP, GR = test(sign(dot(W.transpose(), H_img)), H_tag, Tr_sim, output)

    train_time = float(time.clock())
    print 'train time:', train_time
    #print '---------------begin Test----------------------'
    train_img_time = 0
    train_qa_time = 0
    avg_GP = np.zeros(bit[1])
    avg_GR = np.zeros(bit[1])

    H_img_Test = []
    for cv in range(50):
        #Tst_img, Tst_qa, gd = subsampling(Tst_img, Tst_qa, gd, 20, 0)
        print '---------50 CV----------------'
        print cv
        CV_qa = Tst_qa[:, cv * 200: (cv + 1) * 200]
        CV_gd = gd[:, cv * 200: (cv + 1) * 200]
        #[train_img_time, train_qa_time] = OutSample_Test(Tr_img, Tr_tag, Tr_sim, Tst_img, Tst_qa, W, S, H_img, H_tag, gd, bit, output, parameter)
        [img_time, qa_time, GP, GR, H_img_Test] = OutSample_Test(Tr_img, Tr_tag, Tr_sim, Tst_img, CV_qa, W, S, H_img, H_tag, CV_gd, bit, output, parameter, cv == 0, H_img_Test)
        train_img_time += img_time
        train_qa_time += qa_time
        avg_GP = avg_GP + GP
        avg_GR = avg_GR + GR
        print img_time
        print qa_time
        print 'GP', GP
        print 'GR', GR

    avg_GP = avg_GP / 50.0000
    avg_GR = avg_GR / 50.0000
    output.write('train time: %f, outsample_img time: %f, outsample_qa time: %f' % (train_time, train_img_time, train_qa_time))

    print avg_GP
    print avg_GR
    output.write('\n')
    for i in range(bit[1]):
        output.write('%f, %f \n' % (avg_GP[i], avg_GR[i]))

    output.flush()
    #H_img_Tst = dot()
    #H_img_Tst = np.sign(dot(W.transpose(), H_img_Tst))
    #test(H_img_Tst, H_qa_Tst, gd)

if __name__ == '__main__':

    output = open('Result.txt', 'w')
    bit = [4, 8, 16, 32]
    para = [10, 100, 1000]

    for bit1 in [4]:
        for bit2 in [4]:
        #for alpha in para:
        #for beta in para:
        #    print '------------------------------'
        #    print 'alpha:', alpha
        #    print 'beta:', beta
        #    print '------------------------------'
            output.write('--------------%d, %d---------\n'%(bit1, bit2))
            RaHH([bit1, bit2], output)

    output.close()
