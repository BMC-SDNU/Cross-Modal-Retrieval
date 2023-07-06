function result = calcMeanFMeanPreMeanRecMeanLSRLabel(queryLabel, retrievalLabel, qB, rB, radius)
%% Author: Qing-Yuan Jiang
% Function:
%   calcMapTopkMapTopkPreRecLabel:
%       calculate mean-Fmeasure, mean-Precision, mean-Recall and mean
%       Lookup Success Rate within given hamming radius.
% Input:
%   queryLabel: 0-1 label matrix (numQuery * numLabel) for query set.
%   retrievalLabel: 0-1 label matrix (numQuery * numLabel) for retrieval set. 
%   qB: compressed binary code for query set.
%   rB: compressed binary code for retrieval set.
%   radius: hamming radius.
% Output:
%       result.meanF: map for whole retrieval set
%       result.meanPre: vector. topk-Map for different topk
%       result.meanRec: vector. topk-Precision for different topk
%       result.meanLSR: vector. topk-Recall for different topk

numQuery = size(qB, 1);
F = 0;
Pre = 0;
Rec = 0;
LSR = 0;

for ii = 1: numQuery
    gnd = queryLabel(ii, :) * retrievalLabel' > 0;
    tsum = sum(gnd);
    if tsum == 0
        continue;
    end
    hamm = hammingDist(qB(ii, :), rB);
    totalRelevent = sum(hamm < radius + 0.001);
    ind = hamm < radius + 0.001;
    goodRetrieval = ind.* gnd;
    f = 0;
    pre = 0;
    rec = 0;
    lsr = 0;
    if sum(goodRetrieval) ~= 0
        lsr = 1;
        if totalRelevent ~= 0
            pre = sum(goodRetrieval) / totalRelevent;
            rec = sum(goodRetrieval) / tsum;
            if rec ~= 0
                f = 2 / (1 / rec + 1 / pre);
            end
        end
    end
    
    F = F + f;
    Pre = Pre + pre;
    Rec = Rec + rec;
    LSR = LSR + lsr;
end

result.meanF = F / numQuery;
result.meanPre = Pre / numQuery;
result.meanRec = Rec / numQuery;
result.meanLSR = LSR / numQuery;
end
