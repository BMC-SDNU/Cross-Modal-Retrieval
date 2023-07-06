function [BI_Tr,BT_Tr,BI_Te,BT_Te,traintime,testtime] = main_RFDHLinear(I_tr, T_tr, I_te, T_te, bits, lambda, gamma, maxIter)
% Reference:
% Di Wang, Quan Wang, and Xinbo Gao. 
% Robust and Flexible Discrete Hashing for Cross-Modal Similarity Search. 
% IEEE Transactions on Circuits and Systems for Video Technology, 28(10):2703-2715, 2018.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%

if ~exist('lambda','var')
    lambda = 3;
end
if ~exist('gamma','var')
    gamma = 0.0001;
end
if ~exist('maxIter','var')
    maxIter = 20;
end


traintime1 = cputime;

%% solve objective function
fprintf('start solving RFDH-Linear...\n');
[B, modelI, modelT] = solveRFDHLinear(I_tr', T_tr', bits, lambda, gamma, maxIter);
Yt = B > 0;
BT_Tr = compactbit(Yt');
BI_Tr = BT_Tr;
traintime2 = cputime;
traintime = traintime2 - traintime1;

%% calculate hash codes
testtime1 = cputime;
II = ones(1,size(I_te,1));
R = modelI.R; P = modelI.P; u = modelI.u; %mean = modelI.mean;
% I_te = bsxfun(@minus, I_te, mean);
B1 = R*(P*I_te'-u*II);
B1 = B1 > 0;
R = modelT.R; P = modelT.P; u = modelT.u; %mean = modelT.mean;
% T_te = bsxfun(@minus, T_te, mean);
B2 = R*(P*T_te'-u*II);
B2 = B2 > 0;
BT_Te = compactbit(B2');
BI_Te = compactbit(B1');
testtime2 = cputime;
testtime = testtime2 - testtime1;