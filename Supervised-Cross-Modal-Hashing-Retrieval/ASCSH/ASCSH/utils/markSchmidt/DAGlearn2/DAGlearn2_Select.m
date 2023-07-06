function [w,b,fEvals] = DAGlearn2_Select(method,X,ordered,scoreType,SC,A)
% [w,b,fEvals] = DAGlearn2_Select(method,X,scoreType,SC,A)
%
% Parent selection given using ordering, 
% or Markov blanket selection ignoring ordering
%
% method:
%   'tree' - search over all single parents
%   'enum' - search over all possible subsets
%   'greedy' - greedy search
%   'L1' - search over subsets on L1-regularization path
%
% X : 
%   - nSamples by nNodes {-1,1} data matrix
%
% ordered:
%   0 - Ignore variable ordering and estimate Markov blankets (default)
%   1 - Use variable ordering to do parent selection
%
% scoreType:
%   0 - Bayesian information criterion (default)
%   1 - Validation set likelihood
%
% SC:
%   - nNodes by nNodes {0,1} data matrix containing allowable edges
%   (default: all edges allowed)
%
% A (optional):
%   - nSamples by nVariables {0,1} matrix that is set to 1 when variable
%   was set by intervention

[nSamples,nNodes] = size(X);

if nargin < 6
    A = [];
    if nargin < 5
        SC = ones(nNodes);
        if nargin < 4
            scoreType = 0;
            if nargin < 3
                ordered = 0;
            end
        end
    end
end

if isempty(SC)
    SC = ones(nNodes);
end
SC = SC~=0;

b = zeros(nNodes,1);
w = zeros(nNodes,nNodes);
fEvals = 0;
for n = 1:nNodes
    fprintf('Processing Node %d...',n);
    
    if ordered
        possibleParents = 1:n-1;
        possibleParents = possibleParents(SC(1:n-1,n)~=0);
    else
        possibleParents = [1:n-1 n+1:nNodes];
        possibleParents = possibleParents(SC([1:n-1 n+1:nNodes],n)~=0);
    end
    Xsub = [ones(nSamples,1) X(:,possibleParents)];
    ysub = X(:,n);
    
    if isempty(A)
        intInd = [];
    else
        intInd = A(:,n)~=0;
    end
    
    switch upper(method)
        case 'TREE'
            [wsub,bsub,fEvalssub] = treeSelect(Xsub,ysub,scoreType,nSamples,intInd);
        case 'ENUM'
            [wsub,bsub,fEvalssub] = enumSelect(Xsub,ysub,scoreType,nSamples,intInd);
        case 'GREEDY'
            [wsub,bsub,fEvalssub] = greedySelect(Xsub,ysub,scoreType,nSamples,intInd);
        case 'L1'
            [wsub,bsub,fEvalssub] = L1Select(Xsub,ysub,scoreType,nSamples,intInd);
    end
    b(n) = bsub;
    w(possibleParents,n) = wsub;
    fEvals = fEvals+fEvalssub;
    if ordered
        fprintf('Parents:');
    else
        fprintf('Markov Blanket:');
    end
    for i = find(w(:,n))
        fprintf(' %d',i);
    end
    fprintf('\n');
end

end

%% Score calculation
function [minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd,minScore,minNonZero,minW)
if nargin < 7
    minScore = [];
end

options.Display = 0;
nZ = sum(nonZero);
w = zeros(nZ,1);
if scoreType == 0
    if isempty(intInd)
        funObj = @(w)LogisticLoss(w,X(:,nonZero),y);
    else
        funObj = @(w)LogisticLoss(w,X(~intInd,nonZero),y(~intInd));
    end
    [w,f] = minFunc(funObj,w,options);
    score = 2*f + nZ*log(nSamples);
else
    trainNdx = [1:nSamples]' <= ceil(nSamples/2);
    if isempty(intInd)
        funObj = @(w)LogisticLoss(w,X(trainNdx,nonZero),y(trainNdx));
    else
        funObj = @(w)LogisticLoss(w,X(trainNdx & ~intInd,nonZero),y(trainNdx & ~intInd));
    end
    w = minFunc(funObj,w,options);
    if isempty(intInd)
        score = LogisticLoss(w,X(~trainNdx,nonZero),y(~trainNdx));
    else
        score = LogisticLoss(w,X(~trainNdx & ~intInd,nonZero),y(~trainNdx & ~intInd));
    end
end

if isempty(minScore) || score < minScore
    minW = w;
    minNonZero = nonZero;
    minScore = score;
end
end

%% Optimal Tree
function [w,b,fEvals] = treeSelect(X,y,scoreType,nSamples,intInd)


% Initalize
nVars = size(X,2);
nonZero = false(nVars,1);
nonZero(1) = 1; % Always include bias

% Fit with just bias
[minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd);
fEvals = 1;

% Find variable to add that improves score the most
for v = 2:nVars
    nonZero(v) = 1;
    [minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd,minScore,minNonZero,minW);
    fEvals = fEvals+1;
    nonZero(v) = 0;
end
w = zeros(nVars,1);
w(minNonZero) = minW;
b = w(1);
w = w(2:end);
end

%% Enumeration
function [w,b,fEvals] = enumSelect(X,y,scoreType,nSamples,intInd)

% Initalize
nVars = size(X,2);
nonZero = false(nVars,1);
nonZero(1) = 1; % Always include bias

% Fit with just bias
[minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd);
fEvals = 1;

if nVars == 1
    b = minW;
    w = [];
    return;
end

% Enumerate over all combinations of sparse candidate parents
while 1
    for v = 2:nVars
        if nonZero(v) == 0
            nonZero(v) = 1;
            break;
        else
            nonZero(v) = 0;
        end
    end
    
    if nonZero(v)==0
        break;
    end
    
    [minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd,minScore,minNonZero,minW);
    fEvals = fEvals+1;
    
end
w = zeros(nVars,1);
w(minNonZero) = minW;
b = w(1);
w = w(2:end);
end

%% Greedy
function [w,b,fEvals] = greedySelect(X,y,scoreType,nSamples,intInd)

% Initalize
nVars = size(X,2);
nonZero = false(nVars,1);
nonZero(1) = 1; % Always include bias

% Fit with just bias
[minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd);
fEvals = 1;

% Find variable to add/remove that improves score the most
while 1
    nonZero = minNonZero;
    for v = 2:nVars
        nonZero(v) = ~nonZero(v);
        [minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd,minScore,minNonZero,minW);
        fEvals = fEvals+1;
        nonZero(v) = ~nonZero(v);
    end
    
    if all(nonZero == minNonZero)
        break;
    end
end
w = zeros(nVars,1);
w(minNonZero) = minW;
b = w(1);
w = w(2:end);
end

%% L1
function [w,b,fEvals] = L1Select(X,y,scoreType,nSamples,intInd)

% Initalize
nVars = size(X,2);
nonZero = false(nVars,1);
nonZero(1) = 1; % Always include bias

% Fit with just bias
[minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd);
fEvals = 1;

if nVars == 1
    b = minW;
    w = [];
    return;
end
    
% Set-up L1-regularized objective
if scoreType == 0
    if isempty(intInd)
        funObj = @(w)LogisticLoss(w,X,y);
    else
        funObj = @(w)LogisticLoss(w,X(~intInd,:),y(~intInd));
    end
else
    trainNdx = [1:nSamples]' <= ceil(nSamples/2);
    if isempty(intInd)
        funObj = @(w)LogisticLoss(w,X(trainNdx,:),y(trainNdx));
    else
        funObj = @(w)LogisticLoss(w,X(trainNdx & ~intInd,:),y(trainNdx & ~intInd));
    end
end   
w = zeros(nVars,1);
w(1) = minW;
[f0,g0] = funObj(w);
lambdaMax = max(abs(g0));
increment = lambdaMax/nVars;
lambdaValues = lambdaMax-increment:-increment:0;

options.verbose = 0;
nonZero_old = w~=0;
for lambda = lambdaValues
    
    lambdaVect = lambda*[0;ones(nVars-1,1)];
    w = L1General2_PSSgb(funObj,w,lambdaVect,options);
    
    nonZero = w~=0;
    if any(nonZero ~= nonZero_old)
        [minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd,minScore,minNonZero,minW);
        fEvals = fEvals+1;
    end
    nonZero_old = nonZero;
end
w = zeros(nVars,1);
w(minNonZero) = minW;
b = w(1);
w = w(2:end);
end