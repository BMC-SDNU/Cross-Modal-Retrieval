%% Generate Data
clear all
rand('state',0);randn('state',0);
nSamples = 20000;
scoreType = 0; % 0 for BIC, 1 for validation
interv = 1; % 0 for observational, 1 for interventional
ordered = 1; % 0 for parent selection given ordering, 1 for Markov blanket selection
[X,A,adj] = sampleNetwork('factors',nSamples,0,interv,0);
if ~interv
    A = [];
end
nNodes = size(X,2);
fprintf('Total number of edges = %d\n\n',sum(adj(:)));

% Optimal Tree (to solve the interventional case, you need the Edmonds
% algorithm code and the bioinformatics toolbox)
if (exist('edmonds') && exist('graphshortestpath')) || isempty(A) || all(A(:)==0)
    fprintf('Finding Optimal Tree\n');
    adjMat = DAGlearnG_Tree(X,scoreType,A);
    if interv
        fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
    end
end
pause

% DAG-search w/ no pruning
fprintf('Running DAG-search with no pruning\n');
maxEvals = inf; % Run until a local min is found
adjMat = DAGlearnG_DAGsearch(X,scoreType,[],[],A,maxEvals);
if interv
    fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
end
pause

% Compute SC pruning
fprintf('Running SC pruning\n');
k = 10;
SC = DAGlearnG_Prune_SC(X,k,0,A);
fprintf('Number of edges remaining = %d, false negatives = %d\n\n',sum(SC(:)),sum(SC(:) == 0 & adj(:) == 1));
pause

% DAG-search w/ SC pruning
fprintf('Running DAG-search with SC pruning\n');
adjMat = DAGlearnG_DAGsearch(X,scoreType,SC,[],A,maxEvals);
if interv
    fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
end
pause

% Compute L1MB pruning
fprintf('Running L1MB pruning\n');
L1MB = DAGlearnG_Select('L1',X,0,scoreType,[],A);
fprintf('Number of edges remaining = %d, false negatives = %d\n\n',sum(L1MB(:)~=0),sum(L1MB(:) == 0 & adj(:) == 1));
pause

% DAG-search w/ L1MB pruning
fprintf('Running DAG-search with L1MB pruning\n');
[adjMat,hashTable,scores,funEvals] = DAGlearnG_DAGsearch(X,scoreType,L1MB,[],A,maxEvals);
if interv
    fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
end