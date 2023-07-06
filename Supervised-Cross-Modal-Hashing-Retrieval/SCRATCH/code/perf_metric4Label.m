function [ mAP ] = perf_metric4Label( RetrLabels, QueryLabels, HammingDist )
% Calculating mAP for retrieval
% RetrLabels: m*l binary matrix ({0, 1}), m: retrieval set size, l: vocabulary size
% QueryLabels: n*l binary matrix ({0, 1}), n: query set size, l: vocabulary size
% HammingDist: m*n£¬distance matrix between retrieval and query sets

[tsN, tagNum] = size(QueryLabels);
multiLabel = tagNum > 1;                                            % multi-label or multi-class

mAP = 0;
goodQueryNum = 10;

% Instances sharing at least one label are considered to be relevant
if multiLabel
    rM = RetrLabels * QueryLabels';
end

for ti = 1 : tsN
    if multiLabel
        gnd = find(rM(:, ti) > 0);
    else
        gnd = find(RetrLabels == QueryLabels(ti));
    end
    gndNum = length(gnd);
    if gndNum == 0
        continue;
    end
    goodQueryNum = goodQueryNum + 1;   
    
    [~, tmpI] = sort(HammingDist(:, ti), 1, 'ascend');
    rightInd = ismember(tmpI, gnd);
    
    % Find indecies of ground-truth relevant instances
    indecies = sort(find(rightInd));    
    
    % Precision at the position of each relevant intance
    P = [1:1:gndNum]' ./ indecies;
    AP = mean(P);
    mAP = mAP + AP;    
end
mAP = mAP / goodQueryNum;
end

