function [nodeEnergy,edgeEnergy,edgeWeights,edgeStruct,nRows,nCols] = middlebury_loadDataUGMep(dataDir)

%% Load data

fprintf('dataDir is %s\n',dataDir);
info = load(strcat(dataDir,'/info.txt'));
dataCost = load(strcat(dataDir,'/data.txt'));
smoothCost = load(strcat(dataDir,'/smooth.txt'));
if exist(strcat(dataDir,'/hWeights.txt'))
    fprintf('Edge energies are weighted\n');
    weighted = 1;
    hWeights = load(strcat(dataDir,'/hWeights.txt'));
    vWeights = load(strcat(dataDir,'/vWeights.txt'));
else
    weighted = 0;
end

nStates = info(1);
nNodes = info(2);
nRows = info(3);
nCols = info(4);

assert(nNodes == nRows*nCols);
assert(nNodes*nStates == numel(dataCost));

%% Make edgeStruct

fprintf('Making Adjacency Matrix...\n');
ind1full = zeros(0,1);
ind2full = zeros(0,1);
for c = 1:nCols
    ind1 = [1:nRows-1] + nRows*(c-1);
    ind2 = [2:nRows] + nRows*(c-1);
    ind1full(end+1:end+length(ind1)) = ind1;
    ind2full(end+1:end+length(ind2)) = ind2;
end
for r = 1:nRows
    ind1 = r + nRows*([1:nCols-1]-1);
    ind2 = r + nRows*([2:nCols]-1);
    ind1full(end+1:end+length(ind1)) = ind1;
    ind2full(end+1:end+length(ind2)) = ind2;
end
adj = sparse(ind1full,ind2full,1,nNodes,nNodes);
adj = adj+adj';

fprintf('Making edgeStruct...\n');
edgeStruct = UGM_makeEdgeStruct(adj,nStates);


%% Make energies

fprintf('Making energies...\n');
nodeEnergy = reshape(dataCost,[nNodes nStates]);
edgeEnergy = reshape(smoothCost,[nStates nStates]);
edgeWeights = ones(edgeStruct.nEdges,1);
if weighted
    for e = 1:edgeStruct.nEdges
        pix1 = edgeStruct.edgeEnds(e,1);
        pix2 = edgeStruct.edgeEnds(e,2);
        if pix1 == pix2-1
            % Horizontal Edge
            edgeWeights(e) = hWeights(pix1);
        elseif pix1 == pix2-nRows
            % Vertical Edge
            edgeWeights(e) = vWeights(pix1);
        end
    end
end