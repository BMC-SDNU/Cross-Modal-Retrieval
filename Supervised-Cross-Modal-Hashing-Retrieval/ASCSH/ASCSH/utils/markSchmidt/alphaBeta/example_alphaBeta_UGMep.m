clear all
close all

[nodeEnergy,edgeEnergy,edgeWeights,edgeStruct,nRows,nCols] = middlebury_loadDataUGMep('alphaBeta/data/tsukuba');
[nNodes,nStates] = size(nodeEnergy);

%% Decode by ignoring edges
[junk yInd] = min(nodeEnergy,[],2);
Energy_Ind = UGMep_Energy(yInd,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct.edgeEnds)
figure;imagesc(reshape(yInd,nRows,nCols)');title(sprintf('Independent (Energy = %f)',Energy_Ind));colormap gray;pause(1)

%% Decode with ICM
yICM = UGMep_Decode_ICM(nodeEnergy,edgeEnergy,edgeWeights,edgeStruct);
Energy_ICM = UGMep_Energy(yICM,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct.edgeEnds)
figure;imagesc(reshape(yICM,nRows,nCols)');title(sprintf('ICM (Energy = %f)',Energy_ICM));colormap gray;pause(1)

%% Decode with Alpha-Beta
ySwap = UGMep_Decode_Swap(nodeEnergy,edgeEnergy,edgeWeights,edgeStruct,ones(nNodes,1));
Energy_Swap = UGMep_Energy(ySwap,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct.edgeEnds)
figure;imagesc(reshape(ySwap,nRows,nCols)');title(sprintf('Alpha-Beta (Energy = %f)',Energy_Swap));colormap gray;pause(1)

%% Decode with Alpha-Expansion
yExpand = UGMep_Decode_Expand(nodeEnergy,edgeEnergy,edgeWeights,edgeStruct,ones(nNodes,1));
Energy_Expand = UGMep_Energy(yExpand,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct.edgeEnds)
figure;imagesc(reshape(yExpand,nRows,nCols)');title(sprintf('Alpha-Expansion (Energy = %f)',Energy_Expand));colormap gray;pause(1)

%% Decode with Alpha-Expansion Beta-Shrink (beta = alpha+1)
betaSelect = 5;
yAlphaBeta = UGMep_Decode_ExpandShrink(nodeEnergy,edgeEnergy,edgeWeights,edgeStruct,betaSelect,ones(nNodes,1));
Energy_AlphaBeta = UGMep_Energy(yAlphaBeta,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct.edgeEnds)
figure;imagesc(reshape(yAlphaBeta,nRows,nCols)');title(sprintf('Alpha-Expanion Beta-Shrink with beta=alpha+1 (Energy = %f)',Energy_AlphaBeta));colormap gray;pause(1)

%% Decode with Alpha-Expansion Beta-Shrink
betaSelect = 3;
yAlphaBeta = UGMep_Decode_ExpandShrink(nodeEnergy,edgeEnergy,edgeWeights,edgeStruct,betaSelect,ones(nNodes,1));
Energy_AlphaBeta = UGMep_Energy(yAlphaBeta,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct.edgeEnds)
figure;imagesc(reshape(yAlphaBeta,nRows,nCols)');title(sprintf('Alpha-Expanion Beta-Shrink (Energy = %f)',Energy_AlphaBeta));colormap gray