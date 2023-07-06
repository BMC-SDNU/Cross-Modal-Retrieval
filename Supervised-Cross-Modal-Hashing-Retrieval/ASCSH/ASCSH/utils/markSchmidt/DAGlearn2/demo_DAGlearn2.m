%% Generate Data
clear all
rand('state',0);randn('state',0);
nSamples = 5000;
scoreType = 0; % 0 for BIC, 1 for validation
interv = 1; % 0 for observational, 1 for interventional
ordered = 1; % 0 for parent selection given ordering, 1 for Markov blanket selection
[X,A,adj] = sampleNetwork('factors',nSamples,1,interv,0);
if ~interv
    A = [];
end
nNodes = size(X,2);
fprintf('Total number of edges = %d\n\n',sum(adj(:)));

% Optimal Tree (to solve the interventional case, you need the Edmonds
% algorithm code and the bioinformatics toolbox)
if (exist('edmonds') && exist('graphshortestpath')) || isempty(A) || all(A(:)==0)
    fprintf('Finding Optimal Tree\n');
    adjMat = DAGlearn2_Tree(X,scoreType,A);
    if interv
        fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
    end
end
pause

% DAG-search w/ no pruning
fprintf('Running DAG-search with no pruning\n');
maxEvals = inf; % Run until a local min is found
adjMat = DAGlearn2_DAGsearch(X,scoreType,[],[],A,maxEvals);
if interv
    fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
end
pause

% Compute SC pruning
fprintf('Running SC pruning\n');
k = 10;
SC = DAGlearn2_Prune_SC(X,k,0,A);
fprintf('Number of edges remaining = %d, false negatives = %d\n\n',sum(SC(:)),sum(SC(:) == 0 & adj(:) == 1));
pause

% DAG-search w/ SC pruning
fprintf('Running DAG-search with SC pruning\n');
adjMat = DAGlearn2_DAGsearch(X,scoreType,SC,[],A,maxEvals);
if interv
    fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
end
pause

% Compute L1MB pruning
fprintf('Running L1MB pruning\n');
L1MB = DAGlearn2_Select('L1',X,0,scoreType,[],A);
fprintf('Number of edges remaining = %d, false negatives = %d\n\n',sum(L1MB(:)~=0),sum(L1MB(:) == 0 & adj(:) == 1));
pause

% DAG-search w/ L1MB pruning
fprintf('Running DAG-search with L1MB pruning\n');
[adjMat,hashTable,scores,funEvals] = DAGlearn2_DAGsearch(X,scoreType,L1MB,[],A,maxEvals);
if interv
    fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
end
pause

% Try 10 random initialization, and re-use hash table
nRestarts = 2
for i = 1:2
    if i > 1
        pause
    end
    fprintf('Restarting L1MB-pruned DAG-search and re-using hash (restart %d)\n',i);
    
    % Generate random DAG with half edges added
    adjInit = (rand(nNodes).*triu(ones(nNodes),1)) > .5;
    perm = randperm(nNodes);
    adjInit = adjInit(perm,perm);
    
    % Force random DAG to agree with L1MB pruning
    adjInit = adjInit.*(L1MB ~= 0);
    
    % Run DAG-search, re-using hash
    [adjMat,hashTable,scores,funEvals] = DAGlearn2_DAGsearch(X,scoreType,L1MB,adjInit,A,maxEvals,hashTable);
    if interv
        fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
    end
end