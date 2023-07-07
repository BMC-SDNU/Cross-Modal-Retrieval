function [label_new] = scale2vec(label)
N = size(label, 1);
label_new = sparse(1:N, label, 1);
label_new = full(label_new);
end