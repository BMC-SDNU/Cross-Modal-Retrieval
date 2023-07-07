# Introduction
This is the source code of our TCSVT 2014 paper "Learning Cross-Media Joint Representation with Sparse and Semisupervised Regularization", Please cite the following paper if you use our code.

Xiaohua Zhai, Yuxin Peng, and Jianguo Xiao, "Learning Cross-Media Joint Representation with Sparse and Semisupervised Regularization", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 24, No. 6, pp. 965-978 , Jun. 2014. [[PDF]](http://59.108.48.34/mipl/tiki-download_file.php?fileId=269)

# Usage
Run our script to train and test:
 
    JRL.m

The parameters are as follows:

    I_tr: the feature matrix of image instances for training, dimension : tr_n * d_i
    T_tr: the feature matrix of text instances for training, dimension : tr_n * d_t
    I_te: the feature matrix of image instances for test, dimension : te_n * d_i
    T_te: the feature matrix of text instances for test, dimension : te_n * d_t
    trainCat: the category list of data for training, dimension : tr_n * 1
    testCat: the category list of data for test, dimension : te_n * 1
    gamma: sparse regularization parameter, default: 1000
    sigma: mapping regularization parameter, default: 1000
    lambda: graph regularization parameter, default: 1
    miu: high level regularization parameter, default: 1
    k: kNN parameter, default: 100

The source codes are for Wikipedia dataset, which can be download via: http://www.svcl.ucsd.edu/projects/crossmodal/.

For more information, please refer to our [paper](http://59.108.48.34/mipl/tiki-download_file.php?fileId=269)

# Our Related Work
If you are interested in cross-media retrieval, you can check our recently published overview paper on IEEE TCSVT:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2017.[[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201823)

Welcome to our [Benchmark Website](http://59.108.48.34/mipl/xmedia) and [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.

