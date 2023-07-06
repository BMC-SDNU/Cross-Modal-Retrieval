% Reference:
% Di Wang, Xinbo Gao, Xiumei Wang, Lihuo He, and Bo Yuan. 
% Multimodal Discriminative Binary Embedding for Large-Scale Cross-Modal Retrieval. 
% IEEE Transactions on Image Processing, 25(10):4540-4554, 2016.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
function [Bi_Ir,Bt_Tr,Bi_Ie,Bt_Te,traintime,testtime] = main_MDBE(I_tr, T_tr, I_te, T_te, L, bits, lambda, beta, gamma, maxIter)

if ~exist('lambda','var')
    lambda = 0.5;
end
if ~exist('alpha','var')
    beta = 100;
end
if ~exist('gamma','var')
    gamma = 0.001;
end
if ~exist('maxIter','var')
    maxIter = 20;
end

if isvector(L) 
    L = sparse(1:length(L), double(L), 1); L = full(L);
end

traintime1 = cputime;

mean_I = mean(I_tr, 1);
mean_T = mean(T_tr,1);
I_tr = bsxfun(@minus, I_tr, mean_I);
T_tr = bsxfun(@minus, T_tr, mean_T);  

fprintf('start solving MDBE...\n');
[P1, P2, D] = solveMDBE(I_tr', T_tr', L', lambda, beta, gamma, bits, maxIter);

Yi_tr = sign(L*D')';
Yt_tr = sign(L*D')';
Yi_tr = Yi_tr > 0;
Yt_tr = Yt_tr > 0;

Bt_Tr = compactbit(Yi_tr');
Bi_Ir = compactbit(Yt_tr');

traintime2 = cputime;
traintime = traintime2 - traintime1;

testtime1 = cputime;
I_te = bsxfun(@minus, I_te, mean_I);
T_te = bsxfun(@minus, T_te, mean_T);

Yi_te = sign(P1 * I_te');
Yt_te = sign(P2 * T_te');
Yi_te = Yi_te > 0;
Yt_te = Yt_te > 0;

Bt_Te = compactbit(Yt_te');
Bi_Ie = compactbit(Yi_te');

testtime2 = cputime;
testtime = testtime2 - testtime1;