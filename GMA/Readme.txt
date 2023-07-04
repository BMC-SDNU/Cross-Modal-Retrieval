This package contains modified functions from Abhishek Sharma's implementation of the CVPR 2012 paper
"Generalized Multiview Analysis: A discriminative Latent Space", please cite the work as below.

@inproceedings{gma-cvpr12,
  author    = {Abhishek Sharma and Abhishek Kumar and Hal Daume III and David W Jacobs},
  title     = {Generalized Multiview Analysis: A discriminative Latent Space},
  booktitle = {CVPR},
  year      = {2012},
  pages     = {2160 -- 2167},
}

This package also contains modified functions from Deng Cai's implementation of W and D matrix computations, kindly cite the relevant work as per as 
http://www.cad.zju.edu.cn/home/dengcai/Data/code/constructW.m


NOTE - Read the comments in Newgma.m for using this package in optimal manner, it contains various guidelines for setting parameters. 


Constructing the option structure to compute the matrices W and carry out GMA

========= constructW options ==========================================
** Look into the file constructW for understanding these options and in general refer to the spectral regression codes by Deng Cai **

o.gnd = labTrain1; % label vector for traning, required for constructW
o.Metric = 'Euclidean'; % option for the constructW function
o.NeighborMode = 'Supervised'; % option for constructW function
o.bLDA = 1; % option for constructW function
o.WeightMode = 'Binary'; % options for constrctW function
o.PCA = 1; % Do PCA before GMA
o.PCAthresh = 0.95; % Retain this much energy while doing PCA
o.t = 8; % option for constructW
o.k = 50; % option for constructW
o.intraK = 50; % option for constructW
o.interK = 400; % option for constructW

=============== GMA related options ==================================
o.method = 'cca'; % name of the method to be carried out possible options are 'pca','cca','pls','lda' = gmlda, 'lpp' = gmlpp, 'mfa' = gmmfa etc. for more info see the comments in Newgma
o.Factor = length(trainSub) - 1; % number of factors required from GMA
o.meanMinus = 1; % Set = 1 for subtracting mean in each view, 
o.Dev = 0; % set = 1 for making standard deviation = 1
o.Autopara = 1; % Tune the parameters automatically based on the trace ratio READ the function Newgma's Notes section of comment for more information
o.Mult = [1 1]; % \gamma1 and \gamma2 as per as eqn7 in the paper, note $\gamma = \gamma2/ \gamma1$
o.Lamda = 10; % \alpha in eqn7 in the paper.
o.nPair = 10; % for making random pairs of cross-view samples
o.AlignMode = 2; % Alignmode = 1 for all align; 2 = mean align and 3 = cluster center align % Alignmode = 4 for random alignment
o.NumCluster = 2; % Number of clusters for cluster center alignation
o.ReguAlpha = 1e-6; % regularized term for Bf

