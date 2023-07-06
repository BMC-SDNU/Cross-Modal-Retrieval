%% Load data

clear all
close all

[nodePot,edgePot,edgeStruct,nRows,nCols,maxVal] = middlebury_loadDataUGM('alphaBeta/data/tsukuba');
[nNodes,nStates] = size(nodePot);

%% Decode by ignoring edges
[junk yInd] = max(nodePot,[],2);
Energy_Ind = -maxVal*UGM_LogConfigurationPotential(yInd,nodePot,edgePot,edgeStruct.edgeEnds)
figure;imagesc(reshape(yInd,nRows,nCols)');title(sprintf('Independent (Energy = %f)',Energy_Ind));colormap gray;pause(1);

%% Decode with ICM
yICM = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
Energy_ICM = -maxVal*UGM_LogConfigurationPotential(yICM,nodePot,edgePot,edgeStruct.edgeEnds)
figure;imagesc(reshape(yICM,nRows,nCols)');title(sprintf('ICM (Energy = %f)',Energy_ICM));colormap gray;pause(1);

%% Decode with Alpha-Beta
ySwap = UGM_Decode_AlphaBetaSwap(nodePot,edgePot,edgeStruct,@UGM_Decode_GraphCut,ones(nNodes,1));
Energy_Swap = -maxVal*UGM_LogConfigurationPotential(ySwap,nodePot,edgePot,edgeStruct.edgeEnds)
figure;imagesc(reshape(ySwap,nRows,nCols)');title(sprintf('Alpha-Beta Swap (Energy = %f)',Energy_Swap));colormap gray;pause(1);

%% Decode with Alpha-Expansion
yExpand = UGM_Decode_AlphaExpansion(nodePot,edgePot,edgeStruct,@UGM_Decode_GraphCut,ones(nNodes,1));
Energy_Expand = -maxVal*UGM_LogConfigurationPotential(yExpand,nodePot,edgePot,edgeStruct.edgeEnds)
figure;imagesc(reshape(yExpand,nRows,nCols)');title(sprintf('Alpha-Expansion (Energy = %f)',Energy_Expand));colormap gray;pause(1);

%% Decode with Alpha-Expansion Beta-Shrink (beta = alpha+1)
betaSelect = 5;
yAlphaBeta = UGM_Decode_AlphaExpansionBetaShrink(nodePot,edgePot,edgeStruct,@UGM_Decode_GraphCut,betaSelect,ones(nNodes,1));
Energy_AlphaBeta = -maxVal*UGM_LogConfigurationPotential(yAlphaBeta,nodePot,edgePot,edgeStruct.edgeEnds)
figure;imagesc(reshape(yAlphaBeta,nRows,nCols)');title(sprintf('Alpha-Expanion Beta-Shrink with beta=alpha+1 (Energy = %f)',Energy_AlphaBeta));colormap gray;pause(1);

%% Decode with Alpha-Expansion Beta-Shrink
betaSelect = 3;
yAlphaBeta = UGM_Decode_AlphaExpansionBetaShrink(nodePot,edgePot,edgeStruct,@UGM_Decode_GraphCut,betaSelect,ones(nNodes,1));
Energy_AlphaBeta = -maxVal*UGM_LogConfigurationPotential(yAlphaBeta,nodePot,edgePot,edgeStruct.edgeEnds)
figure;imagesc(reshape(yAlphaBeta,nRows,nCols)');title(sprintf('Alpha-Expanion Beta-Shrink with all beta (Energy = %f)',Energy_AlphaBeta));colormap gray
