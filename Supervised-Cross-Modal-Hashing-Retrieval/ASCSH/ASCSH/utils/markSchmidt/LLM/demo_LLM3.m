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

%% Ising Parameterization

options.param = 'I';

fprintf('Using Ising parameterization with L1-Regularization...');
options.regType = '1';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Ising parameterization with Hierarchical L1-Regularization...');
options.regType = 'H';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Generalized Ising Parameterization

options.param = 'P';

fprintf('Using Generalized Ising parameterization with L1-Regularization...');
options.regType = '1';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Generalized Ising parameterization with Group L1-Regularization...');
options.regType = 'G';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);


fprintf('Using Generalized Ising parameterizatoin with Hierarchical Group L1-Regularization...');
options.regType = 'H';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Full Parameterization

options.param = 'F';

fprintf('Using Full parameterization with L1-Regularization...');
options.regType = '1';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Full parameterization with Group L1-Regularization...');
options.regType = 'G';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);


fprintf('Using Full parameterization with Hierarchical Group L1-Regularization...');
options.regType = 'H';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Canonical Parameterization

options.param = 'C';

fprintf('Using Canonical parameterization with L1-Regularization...');
options.regType = '1';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Canonical parameterization with Hierarchical L1-Regularization...');
options.regType = 'H';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Spectral Parameterization

options.param = 'S';

fprintf('Using Spectral parameterization with L1-Regularization...');
options.regType = '1';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Spectral parameterization with Hierarchical L1-Regularization...');
options.regType = 'H';
options.order = 7;
model = LLM_trainGrow(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);
