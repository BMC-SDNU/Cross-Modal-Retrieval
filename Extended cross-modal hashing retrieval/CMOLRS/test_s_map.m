function [mapk] = test_s_map( smatrix, label_que, label_doc, fd )
% test retrieval using score matrix
% smatrix: score matrix, N_que x N_doc
% calculate agree matrix first

[N_que, N_doc] = size(smatrix);
c = size(label_que, 2);

top_n = [50,N_doc];
[dist, idx] = sort(smatrix, 2, 'descend');
if c == 1 % label is an integer
    agree = bsxfun(@eq, label_que, label_doc(idx(:, :)));
else % label is a vector
    agree = zeros(N_que, N_doc);
    for i = 1 : N_que
        q_label = label_que(i, :);
        dist_label = label_doc(idx(i, :), :);
        agree(i, :)=(sum(dist_label(:,q_label>0),2)>0)';
    end
end

% map
rele = cumsum(agree, 2);
prec = bsxfun(@ldivide, (1:N_doc), rele);
map = bsxfun(@rdivide, cumsum(prec .* agree, 2), max(rele, 1));
mapk = mean(map(:, top_n), 1);

for i = 1 : size(top_n, 2)
    fprintf('%f\t', mapk(i));
end
fprintf('\n');

if nargin > 3 % write to file
    for i = 1 : size(top_n, 2)
        fprintf(fd, '%f\t', mapk(i));
    end
    fprintf(fd, '\n');
end