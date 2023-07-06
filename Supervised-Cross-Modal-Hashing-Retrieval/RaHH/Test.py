#Test function for RaHH algorithm as well as CVH algorithm
#Given the image features, qa features, image hash, qa hash
#And aslo Rpq to mapping the hash code form image domain 
#to QA domain. The global precision, global recall and also MAP
#aacording to references are calcuated. In order to test the performance
#of CVH just set Rpq = eye() to disable it 
#Author: Bo Liu,
#bliuab@cse.ust.hk, 2013.11.2
from numpy import *
from HamDist import *
import pylab as pl

def MAP(dist, gd):
    #Calculate the the mean average percision
    ap = 0

    for p in range(2):

        api = zeros((dist.shape[p], 1), dtype='float')
        for i in range(dist.shape[p]):
            if p == 0:
                rank_i = dist[i, :]
                gd_i = gd[i, :]
            else:
                rank_i = dist[:, i]
                gd_i = gd[:, i]
            #print p
            #print i
            #print rank_i.shape
            #print rank_i
            #print range(len(dist[i]))
            ind = lexsort((range(len(rank_i)), rank_i))
            rank_i = rank_i[ind]
            gd_i = (gd_i)[ind]
            pos_sum = 0
            all_sum = 0

            for j in range(len(rank_i)):

                if gd_i[j] == 1:
                    pos_sum += 1
                    all_sum += 1
                    api[i] += float(pos_sum) / all_sum
                else:
                    all_sum += 1

            api[i] /= all_sum
        ap += api.sum() / dist.shape[p]

    #print ap
    ap /= 2
    return ap


def test(img_hash, qa_hash, groundtruth, output):

    print '-----------------------------------------------'
    dist = zeros((img_hash.shape[1], qa_hash.shape[1]))
    for i in range(img_hash.shape[1]):
        for j in range(qa_hash.shape[1]):
            #print i,j,HamDist(img_hash[:,i],qa_hash[:,j])
            dist[i, j] = HamDist(img_hash[:, i], qa_hash[:, j])


    step = img_hash.shape[0]
    print 'step', step
    dist_threshold = linspace(0, 1, step)
    GP = arange(step, dtype ='f')
    GR = arange(step, dtype = 'f')

    #print dist
    i = 0

    #map_err = MAP(dist, groundtruth)
    #print '-----------MAP----------', map_err
    #print 'dist:', dist.shape
    #print 'gd:', groundtruth.shape

    for thre in dist_threshold:
    #set_printoptions(threshold='nan')

        TP_FP = sum(dist <= thre)
        TP = sum((dist <= thre) * (groundtruth == 1))
        P = sum(groundtruth == 1)

	print 'thre:', thre, 'TP_FP:', TP_FP, 'TP:', TP, 'P:', P

        if isnan(TP):
            TP = 0
        if isnan(TP_FP) or TP_FP == 0:
            TP_FP = 1

        #print TP
        #print TP_FP

        GP[i] = (100.000000 * float(TP)) / float(TP_FP)
        GR[i] = (100.000000 * float(TP)) / float(P)

        #result = 'R:%f bit1: %d, bit2: %d, GP: %f, GR: %f \n' % (thre, img_hash.shape[0], qa_hash.shape[0], GP[i], GR[i])
        result = '(%f, %f),' % (GP[i], GR[i])
        #output.write(result)

        #print 'bit1: %d, bit2: %d' % (img_hash.shape[0], qa_hash.shape[0])
        #print 'GP:', GP[i], 'GR:', GR[i]

        i += 1
        #neg_mean = Tst_sim[dist < thre]
        #pos_mean = Tst_sim[dist >= thre]
        #print 'neg:', mean(neg_mean)
        #print 'pos:', mean(pos_mean)
        #
        #pl.close()
        #pl.plot(neg_mean.ravel(), ones((len(neg_mean.ravel()), 1)), 'go')
        #pl.plot(pos_mean.ravel(), ones((len(pos_mean.ravel()), 1)) + 1, 'ro')
        #pl.savefig(str(i) + '.jpg')
        #
        #i += 1
    
    print 'GP:', GP
    print 'GR:', GR
    #output.write('\n')
    return GP, GR


