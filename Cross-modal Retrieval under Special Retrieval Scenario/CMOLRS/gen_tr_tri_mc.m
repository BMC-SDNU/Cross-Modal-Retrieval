function [ train_data ] = gen_tr_tri_mc( labels, params )
% generate triplets for multi-label dataset
% labels: (c x n) label matrix, n is the number of samples, c is the number of labels
% params.tr_num: number of triplets
train_data = zeros(params.tri_num, 3);
N_tr = size(labels, 1);
for i = 1 : params.tri_num
    q_ind = ceil(rand(1)*N_tr);
    t1_ind = ceil(rand(1)*N_tr);
    t1_rel = sum(bitand(labels(q_ind, :)>=1, labels(t1_ind, :)>=1));
    while 1
        t2_ind = ceil(rand(1)*N_tr);
        t2_rel = sum(bitand(labels(q_ind, :)>=1, labels(t2_ind, :)>=1));
        if t1_rel ~= t2_rel
            break
        end
    end
    if t1_rel > t2_rel
        pos_ind = t1_ind;
        neg_ind = t2_ind;
    else
        pos_ind = t2_ind;
        neg_ind = t1_ind;
    end
    train_data(i, 1) = q_ind;
    train_data(i, 2) = pos_ind;
    train_data(i, 3) = neg_ind;
	
	if mod(i, 10000) == 0
        fprintf('%d\n', i);
    end
end
% save('train_data.mat', 'train_data');
% load('train_data');
end

