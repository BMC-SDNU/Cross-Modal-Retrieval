% Reference:
% Di Wang, Quan Wang, and Xinbo Gao. 
% Robust and Flexible Discrete Hashing for Cross-Modal Similarity Search. 
% IEEE Transactions on Circuits and Systems for Video Technology, 28(10):2703-2715, 2018.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
clc;clear 
load wikiData.mat

%% Calculate the groundtruth
GT = L_te*L_tr';
WtrueTestTraining = zeros(size(L_te,1),size(L_tr,1));
WtrueTestTraining(GT>0)=1;
%% Parameter setting
bit = 32; 
%% Learn RFDH-Linear
[B_I,B_T,tB_I,tB_T] = main_RFDHLinear(I_tr, T_tr, I_te, T_te, bit);
%% Compute mAP
Dhamm = hammingDist(tB_I, B_T)';    
[~, HammingRank]=sort(Dhamm,1);
mapIT = map_rank(L_tr,L_te,HammingRank); 
Dhamm = hammingDist(tB_T, B_I)';    
[~, HammingRank]=sort(Dhamm,1);
mapTI = map_rank(L_tr,L_te,HammingRank); 
map = [mapIT(100),mapTI(100)]
