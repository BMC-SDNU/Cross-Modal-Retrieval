function [nodePot,edgePot,edgeStruct,nRows,nCols,maxVal] = middlebury_loadDataUGM(dataDir)

%% Load data

fprintf('dataDir is %s\n',dataDir);
info = load(strcat(dataDir,'/info.txt'));
dataCost = load(strcat(dataDir,'/data.txt'));

if exist(strcat(dataDir,'/smooth.txt'))
	fprintf('Edge potentials are tied\n');
	tied = 1;
	smoothCost = load(strcat(dataDir,'/smooth.txt'));
else
	fprintf('Edge potentials are untied\n');
	tied = 0;
	hSmooth = load(strcat(dataDir,'/hSmooth.txt'));
	vSmooth = load(strcat(dataDir,'/vSmooth.txt'));
end
if exist(strcat(dataDir,'/hWeights.txt'))
	fprintf('Edge potentials are weighted\n');
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
if tied;assert(nStates^2 == numel(smoothCost)); end

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

%% Make potentials

fprintf('Making potentials...\n');
nodePot = reshape(dataCost,[nNodes nStates]);

if tied
	% All edge potentials are the same, up to a weighting
	edgePot = reshape(smoothCost,[nStates nStates]);
	
	if weighted
		edgeWeights = zeros(edgeStruct.nEdges,1);
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
		
		edgePot = repmat(edgePot,[1 1 edgeStruct.nEdges]);
		for e = 1:edgeStruct.nEdges
			pix1 = edgeStruct.edgeEnds(e,1);
			pix2 = edgeStruct.edgeEnds(e,2);
			if pix1 == pix2-1
				% Horizontal Edge
				edgePot(:,:,e) = edgePot(:,:,e)*hWeights(pix1);
			elseif pix1 == pix2-nRows
				% Vertical Edge
				edgePot(:,:,e) = edgePot(:,:,e)*vWeights(pix1);
			end
		end
	else
		edgePot = repmat(edgePot,[1 1 edgeStruct.nEdges]);
	end
else
	% Potentially different potential on each edge
	
	% Count number of horizontal and vertical edges
	nHorzEdges = 0;
	nVertEdges = 0;
	for e = 1:edgeStruct.nEdges
		pix1 = edgeStruct.edgeEnds(e,1);
		pix2 = edgeStruct.edgeEnds(e,2);
		if pix1 == pix2-1
			% Horizontal Edge
			nHorzEdges = nHorzEdges+1;
		elseif pix1 == pix2-nRows
			% Vertical Edge
			nVertEdges = nVertEdges+1;
		end
	end
	hSmooth = reshape(hSmooth,[nStates nStates nHorzEdges]);
	vSmooth = reshape(vSmooth,[nStates nStates nVertEdges]);
	
	horzEdge = 1;
	vertEdge = 1;
	edgePot = zeros(nStates,nStates,edgeStruct.nEdges);
	for e = 1:edgeStruct.nEdges
		pix1 = edgeStruct.edgeEnds(e,1);
		pix2 = edgeStruct.edgeEnds(e,2);
		if pix1 == pix2-1
			% Horizontal Edge
			edgePot(:,:,e) = hSmooth(:,:,horzEdge);
			horzEdge = horzEdge+1;
		elseif pix1 == pix2-nRows
			% Vertical Edge
			edgePot(:,:,e) = vSmooth(:,:,vertEdge);
			vertEdge = vertEdge+1;
		end
	end
end

% Exponentiate energies to make potentials, after normalizing to avoid huge values
maxVal = max([nodePot(:);edgePot(:)]);
nodePot = exp(-nodePot/maxVal);
edgePot = exp(-edgePot/maxVal);