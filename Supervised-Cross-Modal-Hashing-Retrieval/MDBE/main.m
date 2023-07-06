% Reference:
% Di Wang, Xinbo Gao, Xiumei Wang, Lihuo He, and Bo Yuan. 
% Multimodal Discriminative Binary Embedding for Large-Scale Cross-Modal Retrieval. 
% IEEE Transactions on Image Processing, 25(10):4540-4554, 2016.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
clc;clear 
load wikiData_deep.mat
X = bsxfun(@minus,I_tr, mean(I_tr,1));
X_te = bsxfun(@minus,I_te, mean(I_tr,1));
low_dim = 128; 
[U, ~] =  PCA(X);
PX = U(:,1:low_dim);
I_tr = X * PX;
I_te = X_te * PX;

%% Calculate the groundtruth
GT = L_te*L_tr';
WtrueTestTraining = zeros(size(L_te,1),size(L_tr,1));
WtrueTestTraining(GT>0)=1;
%% Parameter setting
bit = 32; 
%% Learn MDBE
[B_I,B_T,tB_I,tB_T] = main_MDBE(I_tr, T_tr, I_te, T_te, L_tr, bit);
%% Compute mAP
Dhamm = hammingDist(tB_I, B_T)';    
[~, HammingRank]=sort(Dhamm,1);
mapIT = map_rank(L_tr,L_te,HammingRank); 
Dhamm = hammingDist(tB_T, B_I)';    
[~, HammingRank]=sort(Dhamm,1);
mapTI = map_rank(L_tr,L_te,HammingRank); 
map = [mapIT(100),mapTI(100)];
