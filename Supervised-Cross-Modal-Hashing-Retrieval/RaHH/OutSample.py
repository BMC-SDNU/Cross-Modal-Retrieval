from numpy import *
from cvh import *
from loss_func import *
from RaHH import *
import time
#Author: Bo Liu, bliuab@cse.ust.hk
#Date: 2013.11.8
#References:
#Ou, M., Cui, P., Wang, F., Wang, J., Zhu, W., & Yang, S. (2013). Comparing apples to oranges: a scalable solution with heterogeneous hashing.
#Paper presented at the Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining.
#Given the trained statistics, transforming matrix, Hash code. And new data.
#Efficiently train the hash code of new out of samples.

def OutSample_Test(Tr_img, Tr_tag, Tr_sim, Tst_img, Tst_qa, W, S, H_img_Tr, H_tag_Tr, gd, bit, output, parameter, imagehash_cal, imagehash):

    #x: the new n data sample
    #img_fea: the features of images
    #rp: homogeneous similarity. Intuitively, it is assume to be [0...1..0] here without consideration of homogeneous similarity
    #rpq: the heterogeneous similarity. import component for training here.
    #W: trained transformation matrix to mapping the hash code between different domain.
    #H_img_Tr: trained Hash code for existed image
    #S: trained statistics to accelerate training.
    #Rpq: exist heterogeneous similarity to build a new heterogeneous similarity

    #Initialize hash based on CVH
    test_start = time.clock()
    print 'test_start:', test_start

    mp = Tr_img.shape[1]
    mq = Tr_tag.shape[1]
    #rp = Tr_img.shape[0]
    #rq = Tr_tag.shape[0]

    Al_img = hstack((Tr_img, Tst_img))
    Al_tag = hstack((Tr_tag, Tst_qa))

    #partial similarity
    img_sim = vstack((Tr_sim, zeros((Tst_img.shape[1], Tr_sim.shape[1]))))
    tag_sim = hstack((Tr_sim, zeros((Tr_sim.shape[0], Tst_qa.shape[1]))))

    [H_Al_img, qa_nouse, W_nouse, S_nouse, Rp_Al_img, Rq_Al_img, A_img_nouse, A_qa_nouse] = initialize(Al_img, Tr_tag, img_sim, bit)
    H_Al_img = hstack((H_img_Tr, H_Al_img[:, mp::]))

    #TO avoid duplicate calculation of image's hash code
    if imagehash_cal:
        [H_img_query, H_qa_Tst, W_Tst, S_Tst] = train(Al_img, Tr_tag, H_Al_img, H_tag_Tr, S, W, img_sim, Rp_Al_img, Rq_Al_img, True, mp, 0, parameter)
        H_img_Tst = H_img_query[:, mp::]
    else:
        H_img_Tst = imagehash

    train_img = time.clock()
    train_img_time = float(train_img) - float(test_start)

    [img_nouse, H_Al_tag, W_nouse, S_nose, Rp_Al_tag, Rq_Al_tag, A_img_nouse, A_qa_nouse] = initialize(Tr_img, Al_tag, tag_sim, bit)
    H_Al_tag = hstack((H_tag_Tr, H_Al_tag[:, mq::]))

    [H_img_nouse, H_qa_Tst, W_Tst, S_Tst] = train(Tr_img, Al_tag, H_img_Tr, H_Al_tag, S, W, tag_sim, Rp_Al_tag, Rq_Al_tag, True, 0, mq, parameter)
    H_qa_Tst = H_qa_Tst[:, mq::]

    train_qa_time = float(time.clock()) - float(train_img)

    #print '---------------Begin Performance Evaluation----------------'
    H_img_mapped = sign(dot(W.transpose(), H_img_Tst))
    GP, GR = test(H_img_mapped, H_qa_Tst, gd, output)

    return [train_img_time, train_qa_time, GP, GR, H_img_Tst]
