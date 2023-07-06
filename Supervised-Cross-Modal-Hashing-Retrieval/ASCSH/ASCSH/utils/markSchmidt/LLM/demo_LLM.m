clear all
close all
randn('state',0);
rand('state',0);
useMex = 1; % Set to 0 to only use Matlab files

%% Generate a synthetic data set X
fprintf('Generating Synthetic data...\n');
nTrain = 10000;
nTest = 10000;
nNodes = 10;
nStates = 2;
edgeProbs = [.15 .02 .01 0 0 0];
param = 'F';
X = LLM_generate(nTrain+nTest,nNodes,nStates,edgeProbs,param,useMex);
Xtrain = X(1:nTrain,:);
Xtest = X(nTrain+1:end,:);

%% Set regularization parameter 
% (normally you would search for a good value)
lambda = 10;
options.lambda = lambda;
options.verbose = 0; % Turn off verbose output of optimizer
options.infer = 'exact';
testInfer = 'exact';
options.useMex = useMex;
options.param = param;

%% Pairwise Model with L2-Regularization

fprintf('Using Pairwise restriction with L2-Regularization...');
options.regType = '2';
options.order = 2;
model = LLM_trainFull(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Pairwise Model with Group L1-Regularization

fprintf('Using Pairwise restriction with Group L1-Regularization...');
options.regType = 'G';
options.order = 2;
model = LLM_trainFull(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Threeway Model with L2-Regularization

fprintf('Using Threeway restriction with L2-Regularization...');
options.regType = '2';
options.order = 3;
model = LLM_trainFull(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Threeway Model with Group L1-Regularization

fprintf('Using Threeway restriction with Group L1-Regularization...');
options.regType = 'G';
options.order = 3;
model = LLM_trainFull(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Hierarchical Model with Hierarchical Group L1-Regularization

fprintf('Using Hierarchical restriction with Hierarchical L1-Regularization...');
options.regType = 'H';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);
