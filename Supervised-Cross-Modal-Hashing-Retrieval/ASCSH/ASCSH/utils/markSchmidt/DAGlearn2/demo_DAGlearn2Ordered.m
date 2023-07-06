%% Generate Data
clear all
rand('state',0);randn('state',0);
nSamples = 5000;
scoreType = 0; % 0 for BIC, 1 for validation
interv = 1; % 0 for observational, 1 for interventional
ordered = 1; % 1 for parent selection given ordering, 0 for Markov blanket selection
[X,A,adj] = sampleNetwork('factors',nSamples,1,interv,0);
if ~interv
    A = [];
end
nNodes = size(X,2);
fprintf('Total number of edges = %d\n\n',sum(adj(:)));

% Find best tree consistent with ordering
fprintf('Running tree method\n');
Tree = DAGlearn2_Select('tree',X,ordered,scoreType,[],A);
adjMat = Tree ~= 0;
fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));

% Exhaustive enumeration with sparse candidate pruning
fprintf('Running enumeration method with sparse candidate pruning\n');
k = 7;
SC = DAGlearn2_Prune_SC(X,k,ordered,A);
Enum = DAGlearn2_Select('enum',X,ordered,scoreType,SC,A);
adjMat = Enum ~= 0;
fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));

% Greedy Search
fprintf('Running greedy method\n');
Greedy = DAGlearn2_Select('greedy',X,ordered,scoreType,[],A);
adjMat = Greedy ~= 0;
fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));

% L1-Regularization Path
fprintf('Running L1-regularization method\n');
L1 = DAGlearn2_Select('L1',X,ordered,scoreType,[],A);
adjMat = L1 ~= 0;
fprintf('Number of errors = %d, false positives = %d, false negatives = %d\n\n',sum(adjMat(:)~=adj(:)),sum(adjMat(:)==1 & adj(:)==0),sum(adjMat(:)==0 & adj(:) == 1));
