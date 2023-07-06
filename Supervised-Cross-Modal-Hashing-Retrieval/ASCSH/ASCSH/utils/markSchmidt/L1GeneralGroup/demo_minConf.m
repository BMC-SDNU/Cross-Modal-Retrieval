%% Generate Some Synthetic Data
clear all

nInstances = 200;
nVars = 250;
nTargets = 10;
sparsityFactor = .5;
flipFactor = .1;
X = [ones(nInstances,1) randn(nInstances,nVars-1)];
W = diag([0;rand(nVars-1,1) < sparsityFactor])*randn(nVars,nTargets);
Y = sign(X*W);
flipPos = rand(nInstances*nTargets,1) < flipFactor;
Y(flipPos) = -Y(flipPos);
        
%% Set up optimization problem
W_init = zeros(nVars,nTargets);
W_init = W_init(:);

% Set up groups 
% (group 0 is not regularzed, we make the bias variables belong to group 0)
groups = repmat([0:nVars-1]',1,nTargets);
groups = groups(:);
nGroups = max(groups);

lambda = 20;
lambdaVect = lambda*ones(nGroups,1); % We have (nVars-1) groups

funObj = @(W)SimultaneousLogisticLoss(W,X,Y);

%% Set Optimization Options
gOptions.maxIter = 2000;
gOptions.verbose = 2; % Set to 0 to turn off output
gOptions.corrections = 10; % Number of corrections to store for L-BFGS methods
gOptions.norm = 2; % Set to inf to use infinity norm

%% Run Solvers for L2-norm

fprintf('\nSpectral Projected Gradient\n');
options = gOptions;
w = L1GeneralGroup_Auxiliary(funObj,W_init,lambdaVect,groups,options);
Wspg = reshape(w,nVars,nTargets);
pause;

fprintf('\nOptimal Projected Gradient\n');
options = gOptions;
options.method = 'opg';
options.L = 1/nInstances;
w = L1GeneralGroup_Auxiliary(funObj,W_init,lambdaVect,groups,options);
Wopg = reshape(w,nVars,nTargets);
pause;

fprintf('\nProjected Quasi-Newton\n');
options = gOptions;
options.method = 'pqn';
w = L1GeneralGroup_Auxiliary(funObj,W_init,lambdaVect,groups,options);
Wpqn = reshape(w,nVars,nTargets);
pause;

fprintf('\nBarzilai-Borwein Soft-Threshold\n');
options = gOptions;
w = L1GeneralGroup_SoftThresh(funObj,W_init,lambdaVect,groups,options);
Wbbst = reshape(w,nVars,nTargets);
pause;

fprintf('\nQuasi-Newton Soft-Threshold\n');
options = gOptions;
options.method = 'qnst';
w = L1GeneralGroup_SoftThresh(funObj,W_init,lambdaVect,groups,options);
Wbbst = reshape(w,nVars,nTargets);
pause;