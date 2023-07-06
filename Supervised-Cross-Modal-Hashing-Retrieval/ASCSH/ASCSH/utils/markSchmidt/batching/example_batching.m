clear all

%% Load Data
maxIter = 3;

lambda = 1;
[Xnode,Xedge,Y,edgeStruct,nodeMap,edgeMap,nRows,nCols] = prepareBinaryDenoising;
nInstances = size(Y,1);
w_init = UGM_initWeights(nodeMap,edgeMap);
funObj = @(w)penalizedL2(w,@UGM_CRF_PseudoNLL,lambda,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct);
funObjBatch = @(w,batch)penalizedL2(w,@UGM_CRF_PseudoNLL,lambda*length(batch)/nInstances,Xnode(batch,:,:),Xedge(batch,:,:),Y(batch,:,:),nodeMap,edgeMap,edgeStruct);

% Show examples of noisy images
figure;
for i = 1:4
	subplot(2,2,i);
	imagesc(-reshape(Xnode(i,2,:),nRows,nCols));
	colormap gray
end
suptitle('Examples of Noise-Corrupted Images');
fprintf('(paused)\n');
pause

%% Evaluate with random parameters

figure;
fprintf('Testing random parameters...\n');
w = randn(size(w_init));
for i = 1:4
	subplot(2,2,i);
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
	nodeBel = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
	imagesc(-reshape(nodeBel(:,2),nRows,nCols));
	colormap gray
end
suptitle('Loopy BP node marginals with random parameters');
fprintf('(paused)\n');
pause

%% L-BFGS
fprintf('Estimating parameters with deterministic algorithm...\n');
options.maxIter = maxIter;
options.maxFunEvals = maxIter;
options.Display = 'Full';
options.Corr = 10;
w = minFunc(funObj,w_init,options);

figure;
for i = 1:4
	subplot(2,2,i);
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
	nodeBel = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
	imagesc(-reshape(nodeBel(:,2),nRows,nCols));
	colormap gray
end
suptitle(sprintf('Loopy BP node marginals after %d passes of deterministic algorithm',maxIter));
fprintf('(paused)\n');
pause

%% Stochastic Gradient
fprintf('Estimating parameters with stochastic algorithm...\n');
stepSize = 1e-4;
w = w_init;
for iter = 1:maxIter*nInstances
	i = ceil(rand*nInstances);
	[f,g] = funObjBatch(w,i);
	fprintf('Iter = %d of %d (fsub = %f)\n',iter,maxIter*nInstances,f);
	w = w - stepSize*g;
end

figure;
for i = 1:4
	subplot(2,2,i);
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
	nodeBel = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
	imagesc(-reshape(nodeBel(:,2),nRows,nCols));
	colormap gray
end
suptitle(sprintf('Loopy BP node marginals after %d passes of stochastic algorithm',maxIter));
fprintf('(paused)\n');
pause

%% Hybrid
fprintf('Estimating parameters with hybrid algorithm...\n');
options.t0 = 1e-8;
options.gamma = 1.1;
options.delta = 1;
w = batchingLBFGS(funObjBatch,w_init,nInstances,options);

figure;
for i = 1:4
	subplot(2,2,i);
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
	nodeBel = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
	imagesc(-reshape(nodeBel(:,2),nRows,nCols));
	colormap gray
end
suptitle(sprintf('Loopy BP node marginals after %d passes of hybrid algorithm',maxIter));
fprintf('(paused)\n');
pause