clear all

%% Generate Synthetic Data
nInstances = 10000;
nVars = 1000;
X = randn(nInstances,nVars);
w = randn(nVars,1);
y = sign(X*w + randn(nInstances,1));

%% Set Regularization Parameter and Initial Guess
lambda = 1;
w0 = zeros(nVars,1);

%% Specify number of effective passes through the data
maxIter = 25;

%% L-BFGS
fprintf('Running L-BFGS\n');
options.maxFunEvals = maxIter;
options.Corr = 10;
funObj = @(w)penalizedL2(w,@SSVMLoss,lambda,X,y);
w = minFunc(funObj,w0,options);
f_minFunc = funObj(w);

%% Stochastic Gradient Descent
% fprintf('Running SGD\n');
% stepSize = 1e-4;
% w = w0;
% for iter = 1:maxIter*nInstances
% 	i = ceil(rand*nInstances);
% 	[f,g] = penalizedL2(w,@SSVMLoss,lambda/nInstances,X(i,:),y(i));
% 	if mod(iter,nVars)==0
% 		fprintf('Iter = %d of %d (fsub = %f)\n',iter,maxIter*nInstances,f);
% 	end
% 	w = w - stepSize*g;
% end
% funObj(w)
% pause

%% Hybrid (Growing Batch)
fprintf('Running L-BFGS Growing-Batch Strategy\n');
Xt = X';
funObjBatch = @(w,batch)penalizedL2(w,@SSVMLossBatch_Xtranspose,lambda*length(batch)/nInstances,Xt,y,batch);
options.maxIter = maxIter;
options.t0 = 1e-8;
options.gamma = 1.1;
options.delta = 1;
options.fullObj = funObj;
w = batchingLBFGS(funObjBatch,w0,nInstances,options);
f_grow = funObj(w);

%% Hybrid (Growing Batch and Pruning)
fprintf('Running L-BFGS Growing-Batch and Shrinking Strategy\n');
w = batchingL2SVM(Xt,y,lambda,w0,options);
f_growShrink = funObj(w);

%% Display results
fprintf('\n\n');
fprintf('%f : function value with minFunc after %d effective passes through data\n',f_minFunc,maxIter);
fprintf('%f : function value with growing-batch strategy\n',f_grow);
fprintf('%f : function value with growing-batch and shrinking strategy\n',f_growShrink);

