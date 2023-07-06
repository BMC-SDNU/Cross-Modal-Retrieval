function [w,fEvals] = DAGlearnG_Select(method,X,ordered,scoreType,SC,A)
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
%   - nSamples by nNodes data matrix (columns must be zero-mean)
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
    Xsub = X(:,possibleParents);
    ysub = X(:,n);
    
    if isempty(A)
        intInd = [];
    else
        intInd = A(:,n)~=0;
    end
    
    switch upper(method)
        case 'TREE'
            [wsub,fEvalssub] = treeSelect(Xsub,ysub,scoreType,nSamples,intInd);
        case 'ENUM'
            [wsub,fEvalssub] = enumSelect(Xsub,ysub,scoreType,nSamples,intInd);
        case 'GREEDY'
            [wsub,fEvalssub] = greedySelect(Xsub,ysub,scoreType,nSamples,intInd);
        case 'L1'
            [wsub,fEvalssub] = L1Select(Xsub,ysub,scoreType,nSamples,intInd);
    end
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
% Right now does not store/update Cholesky and always solves from scratch
function [minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd,minScore,minNonZero,minW)

if nargin < 7
    minScore = [];
end

if scoreType == 0
    if isempty(intInd)
        Xtrain = X(:,nonZero);
        ytrain = y;
    else
        Xtrain = X(~intInd,nonZero);
        ytrain = y(~intInd);
    end
    w = Xtrain\ytrain;
    n = size(Xtrain,1);
    sigma2 = sum((Xtrain*w - ytrain).^2)/n;
    nll = n*log(sqrt(sigma2)) + (n/2)*log(2*pi) + (norm(Xtrain*w-ytrain)^2)/(2*sigma2);
    score = 2*nll + length(w)*log(nSamples);
else
    trainNdx = [1:nSamples]' <= ceil(nSamples/2);
    if isempty(intInd)
        Xtrain = X(trainNdx,nonZero);
        ytrain = y(trainNdx);
    else
        Xtrain = X(trainNdx & ~intInd,nonZero);
        ytrain = y(trainNdx & ~intInd);
    end
    w = Xtrain\ytrain;
    n = size(Xtrain,1);
    sigma2 = sum((Xtrain*w - ytrain).^2)/n;
    if isempty(intInd)
        Xtest = X(~trainNdx,nonZero);
        ytest = y(~trainNdx);
    else
        Xtest = X(~trainNdx & ~intInd,nonZero);
        ytest = y(~trainNdx & ~intInd);
    end
    n = size(Xtest,1);
    score = n*log(sqrt(sigma2)) + (n/2)*log(2*pi) + (norm(Xtest*w-ytest)^2)/(2*sigma2);
end

if isempty(minScore) || score < minScore
    minW = w;
    minNonZero = nonZero;
    minScore = score;
end
end

%% Optimal Tree
function [w,fEvals] = treeSelect(X,y,scoreType,nSamples,intInd)

% Initalize
nVars = size(X,2);
nonZero = false(nVars,1);

% Fit no variables
[minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd);
fEvals = 1;

% Find variable to add that improves score the most
for v = 1:nVars
    nonZero(v) = 1;
    [minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd,minScore,minNonZero,minW);
    fEvals = fEvals+1;
    nonZero(v) = 0;
end
w = zeros(nVars,1);
w(minNonZero) = minW;
end

%% Enumeration
function [w,fEvals] = enumSelect(X,y,scoreType,nSamples,intInd)

% Initalize
nVars = size(X,2);
nonZero = false(nVars,1);

% Fit with no parents
[minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd);
fEvals = 1;

if nVars == 0
    w = [];
    return;
end

% Enumerate over all combinations of sparse candidate parents
while 1
    for v = 1:nVars
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
end

%% Greedy
function [w,fEvals] = greedySelect(X,y,scoreType,nSamples,intInd)

% Initalize
nVars = size(X,2);
nonZero = false(nVars,1);

% Fit with no parents
[minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd);
fEvals = 1;

% Find variable to add/remove that improves score the most
while 1
    nonZero = minNonZero;
    for v = 1:nVars
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
end

%% L1
function [w,fEvals] = L1Select(X,y,scoreType,nSamples,intInd)

% Initalize
nVars = size(X,2);
nonZero = false(nVars,1);

% Fit with no parents
[minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd);
fEvals = 1;

if nVars == 0
    w = [];
    return;
end

% Set-up L1-regularized objective
if scoreType == 0
    if isempty(intInd)
        Xtrain = X;
        ytrain = y;
    else
        Xtrain = X(~intInd,:);
        ytrain = y(~intInd);
    end
else
    trainNdx = [1:nSamples]' <= ceil(nSamples/2);
    if isempty(intInd)
        Xtrain = X(trainNdx,:);
        ytrain = y(trainNdx);
    else
        Xtrain = X(trainNdx & ~intInd,:);
        ytrain = y(trainNdx & ~intInd);
    end
end
method = 'lasso';stop = 0;usegram = 0;gram = []; trace = 0;
W = lars(Xtrain,ytrain,'lasso',stop,usegram,gram,trace)';
for i = 2:size(W,2);
    nonZero = W(:,i)~=0;
    [minScore,minNonZero,minW] = computeScore(nonZero,X,y,scoreType,nSamples,intInd,minScore,minNonZero,minW);
    fEvals = fEvals+1;
end
w = zeros(nVars,1);
w(minNonZero) = minW;
end