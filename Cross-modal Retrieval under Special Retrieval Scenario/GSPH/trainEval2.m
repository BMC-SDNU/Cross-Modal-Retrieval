function [ map ] = trainEval2( labels, codes1,codes2 )
% evaluation for learnt hash codes on a training set
% labels: n*l binary matrix, n: training set size, l: vocabulary size
% codes: n*c binary hash code matrix ({-1, 1}), n: training set size, c: hash code legnth

% for {-1, 1} hash codes, Hamming Distance is proportional Euclidean Distance
dist = codes1 * codes2';
dd = diag(dist);
[n, c] = size(codes1);
dist = repmat(dd, 1, n) + repmat(dd', n, 1) - 2 * dist;

map = perf_metric4Label(labels, labels, dist);
end

