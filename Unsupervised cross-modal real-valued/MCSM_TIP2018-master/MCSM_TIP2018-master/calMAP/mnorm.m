function [tr_norm] = mnorm(tr)

maxtr = (max(tr'))';
mintr = (min(tr'))';
tr_norm = (tr - repmat(mintr, 1, size(tr,2))) ./ (repmat((maxtr - mintr), 1, size(tr,2)));
tr_norm(find(isnan(tr_norm))) = 0;
