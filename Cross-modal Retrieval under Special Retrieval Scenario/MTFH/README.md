# MTFH
Matlab demo code for "MTFH: A Matrix Tri-Factorization Hashing Framework for Efficient Cross-Modal Retrieval", to appear in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi:10.1109/TPAMI.2019.2940446

![image](https://github.com/starxliu/MTFH/blob/master/data/framework.jpg)

The framework of MTFH algorithm.

############### Information ################

Matlab demo code for "MTFH: A Matrix Tri-Factorization Hashing Framework for Efficient Cross-Modal Retrieval" 

Authors: Xin Liu, Zhikai Hu, Haibin Ling, and Yiu-ming Cheung;
Contact: xliu[at]hqu.edu.cn; zkhu[at]hqu.edu.cn


This code uses some public software packages by the 3rd party applications, and is free for educational, academic research and non-profit purposes. Not for commercial/industrial activities. If you use/adapt our code in your work (either as a stand-alone tool or as a component of any algorithm), you need to appropriately cite our work.



################ Tips ################
1. To run a demo, conduct the following command:
        MTFH_demo.m

* If you have got any question, please do not hesitate to contact us.
* Bugs are also welcome to be reported.

################ Contents ################

This package contains cleaned up codes for the MTFH, including:

MTFH_demo.m: test example on public Wiki dataset

bitCompact.m: function to compute the compact hash code matrix

hammingDist.m: function to compute the hamming distance between two sets

kernelMatrix.m: function to compute a kernel matrix

perf_metric4Label.m: function to compute mAP for retrieval evaluation

rndsolveUHV2.m: function to optimize the solution of MTFH


################ Citation and Thanks  ################

Xin Liu, Zhikai Hu, Haibin Ling, and Yiu-ming Cheung; MTFH: A Matrix Tri-Factorization Hashing Framework for Efficient Cross-Modal Retrieval, IEEE Transactions on Pattern Analysis and Machine Intelligence, doi:10.1109/TPAMI.2019.2940446

