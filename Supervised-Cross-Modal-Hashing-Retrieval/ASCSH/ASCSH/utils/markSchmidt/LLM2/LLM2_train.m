function [model] = LLM2_train(X,options,model)

[lambda,param,regType,useMex,infer,edges,verbose] = myProcessOptions(options,'lambda',1,'param','F','regType','2','useMex',1,'infer','exact','edges',[],'verbose',1);

[nSamples,nNodes] = size(X);
nStates = max(X(:));

%% Initialize Edges
if isempty(edges) % Use all edges
    edges = zeros(0,2);
    for n1 = 1:nNodes
        for n2 = n1+1:nNodes
            edges(end+1,:) = [n1 n2];
        end
    end
end
nEdges = size(edges,1);

%% Initialize Weights
[w1,w2] = LLM2_initWeights(param,nNodes,nStates,edges);
if nargin < 3
    % Cold-start with all parameters zero
    w = [w1(:);w2(:)];
else
    % Warm-start at previous model
    w = model.w;
end

%% Convert everything to int32
X = int32(X);
nStates = int32(nStates);
edges = int32(edges);

%% Compute sufficient statistics of data and form loss function
switch infer
    case 'exact'
        if useMex
            [ss1,ss2] = LLM2_suffStatC(param,X,nStates,edges);
        else
            [ss1,ss2] = LLM2_suffStat(param,X,nStates,edges);
        end
        
        funObj = @(w)LLM2_NLL(w,param,nSamples,ss1,ss2,edges,useMex);
    case 'pseudo'
        [Xunique,Xreps] = LLM_unique(X);
        funObj = @(w)LLM2_pseudo(w,param,Xunique,Xreps,nStates,edges,useMex);
    case {'tree','mean','loopy','trbp'}
        [V,E] = UGM_makeEdgeVE(edges,nNodes);
        
        edgeStruct.edgeEnds = edges;
        edgeStruct.V = V;
        edgeStruct.E = E;
        edgeStruct.nStates = repmat(nStates,[nNodes 1]);
        edgeStruct.useMex = useMex;
        edgeStruct.nEdges = size(edges,1);
        edgeStruct.maxIter = 250;
        
        if useMex
            [ss1,ss2] = LLM2_suffStatC(param,X,nStates,edges);
        else
            [ss1,ss2] = LLM2_suffStat(param,X,nStates,edges);
        end
        
        switch infer
            case 'tree'
                inferFunc = @UGM_Infer_Tree;
            case 'mean'
                inferFunc = @UGM_Infer_MeanField;
            case 'loopy'
                inferFunc = @UGM_Infer_LBP;
            case 'trbp'
                inferFunc = @UGM_Infer_TRBP;
                weightType = 2;
        end
        
        if strcmp(infer,'trbp')
            funObj = @(w)LLM2_ugmNLL(w,param,nSamples,ss1,ss2,edgeStruct,inferFunc,weightType);
        else
            funObj = @(w)LLM2_ugmNLL(w,param,nSamples,ss1,ss2,edgeStruct,inferFunc);
        end
end

%% Optimize
if verbose
    options.Display = 'iter';
    optiosn.verbose = 2;
else
options.Display = 0;
options.verbose = 0;
end
switch regType
    case '2' % L2-regularization
        lambdaVect = lambda*[zeros(numel(w1),1);ones(numel(w2),1)];
        regFunObj = @(w)penalizedL2(w,funObj,lambdaVect);
        w = minFunc(regFunObj,w,options);
    case '1' % L1-regularization
        lambdaVect = lambda*[zeros(numel(w1),1);ones(numel(w2),1)];
        w = L1General2_PSSgb(funObj,w,lambdaVect,options);
    otherwise % Group L1-regularization
        switch regType
            case 'G'
                options.norm = 2;
            case 'I'
                options.norm = inf;
            case 'N'
                options.norm = 0;
        end
        options.method = 'pqn';
        g1 = zeros(size(w1));
        g2 = zeros(size(w2));
        groups = makeGroups(param,g1,g2,nEdges);
        lambdaVect = lambda*ones(nEdges,1);
        w = L1GeneralGroup_Auxiliary(funObj,w,lambdaVect,groups,options);
end

%% Make model struct
model.param = param;
model.useMex = useMex;
model.nStates = nStates;
model.w = w;
model.edges = edges;
model.nll = @nll;
model.infer = infer;

end

%% Make groups for Group L1-Regularization
function groups = makeGroups(param,g1,g2,nEdges)
for e = 1:nEdges
    switch param
        case {'C','I','S'}
            g2(e) = e;
        case 'P'
            g2(:,e) = e;
        case 'F'
            g2(:,:,e) = e;
    end
end
groups = [g1(:);g2(:)];
end

%% Function for evaluating nll given data X
function f = nll(model,X,infer)

% Convert everything to int32
nSamples = size(X,1);
X = int32(X);

switch infer
    case 'exact'
        % Compute sufficient statistics of data
        if model.useMex
            [ss1,ss2] = LLM2_suffStatC(model.param,X,model.nStates,model.edges);
        else
            [ss1,ss2] = LLM2_suffStat(model.param,X,model.nStates,model.edges);
        end
        f = LLM2_NLL(model.w,model.param,nSamples,ss1,ss2,model.edges,model.useMex);
    case 'pseudo'
        [Xunique,Xreps] = LLM_unique(X);
        f = LLM2_pseudo(model.w,model.param,Xunique,Xreps,model.nStates,model.edges,model.useMex);
end
end