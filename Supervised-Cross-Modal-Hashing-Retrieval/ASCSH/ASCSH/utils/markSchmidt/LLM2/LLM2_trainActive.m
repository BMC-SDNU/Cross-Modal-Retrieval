function [model] = LLM2_trainActive(X,options,model)

[lambda,param,regType,useMex,verbose,infer] = myProcessOptions(options,'lambda',1,'param','F','regType','H','useMex',1,'verbose',1,'infer','exact');
[nSamples,nNodes] = size(X);
nStates = max(X(:));

%% Initialize Edges and weights
if nargin < 3
    % Cold-start with no edges
    edges = zeros(0,2);
    [w1,w2] = LLM2_initWeights(param,nNodes,nStates,edges);
else
    % Warm-start at previous model
    edges = model.edges;
    [w1,w2] = LLM2_splitWeights(model.w,param,nNodes,nStates,edges);
end

%% Convert everything to int32
X = int32(X);
nStates = int32(nStates);
edges = int32(edges);

for i = 1:10
    
    % Solve with current active set
    [w1,w2] = optimize(param,X,w1,w2,edges,lambda,useMex,regType,infer);
    
    edges_old = edges;
    
    % Update parameters of dense graph
    [edges,w1,w2] = updateEdges(param,w1,w2,edges);
    
    % Prune edges that don't give local improvement
    [edges,w1,w2] = pruneEdges(param,X,lambda,w1,w2,edges,verbose,useMex,regType,infer);
    
    % Check if optimal solution found
    if numel(edges)==numel(edges_old) && all(edges(:) == edges_old(:))
        break;
    end
end
model.w = [w1(:);w2(:)];
model.useMex = useMex;
model.nStates = int32(nStates);
model.edges = int32(edges);
model.nll = @nll;
model.infer = infer;
model.param = param;
end

function [edges,w1,w2] = updateEdges(param,w1,w2,edges)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges = size(edges,1);

% Store weights
weightsHash = java.util.Hashtable;
for e = 1:nEdges
    switch param
        case {'I','C','S'}
            weightsHash.put(num2str(edges(e,:)),w2(e));
        case 'P'
            weightsHash.put(num2str(edges(e,:)),w2(:,e));
        case 'F'
            weightsHash.put(num2str(edges(e,:)),w2(:,:,e));
    end
end

% Construct a graph with all edges
edges = zeros(0,2);
for n1 = 1:nNodes
    for n2 = n1+1:nNodes
        edges(end+1,:) = [n1 n2];
    end
end
nEdges = size(edges,1);

% Fill in parameters of edges from previous problem
[junk,w2] = LLM2_initWeights(param,nNodes,nStates,edges);
for e = 1:nEdges
    key = num2str(edges(e,:));
    if weightsHash.containsKey(key)
        switch param
            case {'I','C','S'}
                w2(e) = weightsHash.get(key);
            case 'P'
                w2(:,e) = weightsHash.get(key);
            case 'F'
                w2(:,:,e) = weightsHash.get(key);
        end
    end
end
end

%%
function [edges,w1,w2] = pruneEdges(param,X,lambda,w1,w2,edges,verbose,useMex,regType,infer)

[nSamples,nNodes] = size(X);
nStates = size(w1,2)+1;
nEdges = size(edges,1);

% Convert everything to int32
X = int32(X);
nStates = int32(nStates);
edges = int32(edges);

if strcmp(infer,'exact')
    % Compute sufficient statistics
    if useMex
        [ss1,ss2] = LLM2_suffStatC(param,X,nStates,edges);
    else
        [ss1,ss2] = LLM2_suffStat(param,X,nStates,edges);
    end
    
    % Evaluate gradient
    [f,g] = LLM2_NLL([w1(:);w2(:)],param,nSamples,ss1,ss2,edges,useMex);
else
    [Xunique,Xreps] = LLM_unique(X);
    [f,g] = LLM2_pseudo([w1(:);w2(:)],param,Xunique,Xreps,nStates,edges,useMex);
end
[g1,g2] = LLM2_splitWeights(g,param,nNodes,nStates,edges);

killedEdges = zeros(0,1);
for e = 1:nEdges
    switch param
        case {'I','C','S'}
            w = w2(e);
        case 'P'
            w = w2(:,e);
        case 'F'
            w = w2(:,:,e);
    end
    if all(w(:) == 0)
        switch param
            case {'I','C','S'}
                g = g2(e);
            case 'P'
                g = g2(:,e);
            case 'F'
                g = g2(:,:,e);
        end
        if regType == '1'
            ng = max(abs(g(:)));
        else
            ng = norm(g(:));
        end
        if ng <= lambda
            killedEdges(end+1,1) = e;
        elseif verbose
            fprintf('Adding edge (%d,%d)\n',edges(e,:));
        end
    end
end
edges(killedEdges,:) = [];
switch param
    case {'I','C','S'}
        w2(killedEdges) = [];
    case 'P'
        w2(:,killedEdges) = [];
    case 'F'
        w2(:,:,killedEdges) = [];
end
end

%%
function [w1,w2] = optimize(param,X,w1,w2,edges,lambda,useMex,regType,infer)

useMex = 1;
[nSamples,nNodes] = size(X);
nStates = size(w1,2)+1;
nEdges = size(edges,1);

w = [w1(:);w2(:)];
nVars = length(w);

% Convert everything to int32
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
        
        edgeStruct.edgeEnds = double(edges);
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

% Solve optimization
options.verbose = 0;
switch regType
    case '1'
        lambdaVect = lambda*[zeros(numel(w1),1);ones(numel(w2),1)];
        w = L1General2_PSSgb(funObj,w,lambdaVect,options);
    case 'G'
        lambdaVect = lambda*ones(nEdges,1);
        g1 = zeros(size(w1));
        g2 = zeros(size(w2));
        groups = makeGroups(param,g1,g2,nEdges);
        w = L1GeneralGroup_Auxiliary(funObj,w,lambdaVect,groups,options);
end
[w1,w2] = LLM2_splitWeights(w,param,nNodes,nStates,edges);
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