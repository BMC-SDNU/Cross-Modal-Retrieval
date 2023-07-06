clear all;
warning off; 
clc;

%% Load data
load nuswide.mat;
fprintf('NUS-WIDE dataset loaded...\n');

%% Parameter setting
run = 1;
map = zeros(run,1);
n_anchors = 1000;
param.bits = 128;
param.batch = 1000;
param.alpha = 1e-5;
param.beta = 1e-1;
param.gamma = 1e-3;
param.rho = 1e-1;

%% Data preparing
Ntrain = size(I_tr,1); 
sample = randsample(Ntrain, n_anchors);

anchorI = I_tr(sample,:);
anchorT = T_tr(sample,:);
sigmaI = 140;
sigmaT = 60;

PhiI = exp(-sqdist(I_tr,anchorI)/(2*sigmaI*sigmaI));
PhiI = [PhiI, ones(Ntrain,1)];
PhiT = exp(-sqdist(T_tr,anchorT)/(2*sigmaT*sigmaT));
PhiT = [PhiT, ones(Ntrain,1)];

Phi_testI = exp(-sqdist(I_te,anchorI)/(2*sigmaI*sigmaI));
Phi_testI = [Phi_testI, ones(size(Phi_testI,1),1)];
Phi_testT = exp(-sqdist(T_te,anchorT)/(2*sigmaT*sigmaT));
Phi_testT = [Phi_testT, ones(size(Phi_testT,1),1)];

Phi_dbI = exp(-sqdist(I_db,anchorI)/(2*sigmaI*sigmaI));
Phi_dbI = [Phi_dbI, ones(size(Phi_dbI,1),1)];
Phi_dbT = exp(-sqdist(T_db,anchorT)/(2*sigmaT*sigmaT));
Phi_dbT = [Phi_dbT, ones(size(Phi_dbT,1),1)];

PhiI = PhiI';
PhiT = PhiT';
Phi_testI = Phi_testI';
Phi_testT = Phi_testT';
Phi_dbI = Phi_dbI';
Phi_dbT = Phi_dbT';

phi_x1 = PhiI; 
phi_x2 = PhiT;
phi_db1 = Phi_dbI; 
phi_db2 = Phi_dbT;
phi_test1 = Phi_testI; 
phi_test2 = Phi_testT;

for j = 1 : run
    %% Training Model
    [W1, W2] = solveFOMH(PhiI, PhiT, L_tr, param, j);
    
    %% When new query samples come
    [B_test, B_db] = queryFOMH(Phi_dbI, Phi_dbT, Phi_testI, Phi_testT, W1, W2, param);
    
    %% Evaluation
    B_db   = compactbit(B_db>0);
    B_test = compactbit(B_test>0);
    Dhamm  = hammingDist(B_db+2, B_test+2);
    [P2]   = perf_metric4Label(L_db, L_te, Dhamm);
    map(j) = P2;
end

fprintf('\n==================================================== mAP at %d-bits:%.4f ====================================================\n\n', param.bits, mean(map));

