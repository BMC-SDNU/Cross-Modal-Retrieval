clear all
close all
randn('state',0);
rand('state',0);
useMex = 1; % Set to 0 to only use Matlab files

%% Generate a synthetic data set X
fprintf('Generating Synthetic data...\n');
nTrain = 5000;
nTest = 5000;
nNodes = 9;
nStates = 2;
edgeProbs = [.2 .15 .1 .05 0 0];
param = 'P';
X = LLM_generate(nTrain+nTest,nNodes,nStates,edgeProbs,param,useMex);
Xtrain = X(1:nTrain,:);
Xtest = X(nTrain+1:end,:);

%% Set sequence of regularization parameters

lambdaValues = 2.^[10:-.25:-8];

%% Set up model parameters
options.regType = 'H';
options.param = 'P';
options.infer = 'pseudo';
testInfer = 'pseudo';
options.useMex = useMex;
options.order = 7;

%% Train and test w/ sequence of regularization parameters
testNLL = inf(length(lambdaValues),1);
minNLL = inf;
for regParam = 1:length(lambdaValues);
    options.lambda = lambdaValues(regParam);
    if regParam == 1
        model = LLM_trainGrow(Xtrain,options);
    else
        model = LLM_trainGrow(Xtrain,options,model);
    end
    testNLL(regParam) = model.nll(model,Xtest,testInfer);
    fprintf('lambda = %f, testNLL = %f, nnz = %d\n',lambdaValues(regParam),testNLL(regParam),nnz(model.w));
    
    if testNLL(regParam) < minNLL
        minLambda = lambdaValues(regParam);
        minNLL = testNLL(regParam);
        minModel = model;
    end
    
    if regParam > max(6,length(lambdaValues)/4) && issorted(testNLL(regParam-5:regParam))
        fprintf('Test NLL increased on 5 consecutive iterations, terminating.\n');
        break;
    end
end
fprintf('Best lambda = %f, nnz = %d\n',minLambda,nnz(minModel.w));
