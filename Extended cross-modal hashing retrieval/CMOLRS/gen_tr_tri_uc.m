function [ train_data ] = gen_tr_tri_uc( classes, params )
% generate triplets for uni-label dataset
% classes: (1 x n) label matrix, n is the number of samples
% params.tr_num: number of triplets
% params.pos: type of positive example. 1 for corresponding, 2 for in the same class
[unused_values, inds] = sort(classes);
classes = classes(inds);
num_classes = length(unique(classes));
class_sizes = zeros(num_classes,1);
class_start = zeros(num_classes,1);
for k=1:num_classes
    class_sizes(k) = sum(classes==k);
    class_start(k) = find(classes==k, 1, 'first');
end

train_data = zeros(params.tri_num, 3);
N_tr = size(classes, 1);
for i = 1 : params.tri_num
    q_ind   = ceil(rand(1)*N_tr);
    class = classes(q_ind);
    if params.pos == 1
        pos_ind = q_ind;
    else
        pos_ind = class_start(class) - 1 + ceil(rand(1)* class_sizes(class));
    end
    neg_ind = ceil(rand(1)*N_tr);
    while  classes(neg_ind) == class
        neg_ind = ceil(rand(1)*N_tr);
    end
    train_data(i, 1) = q_ind;
    train_data(i, 2) = pos_ind;
    train_data(i, 3) = neg_ind;
	
	if mod(i, 10000) == 0
        fprintf('%d\n', i);
    end
end
train_data(:,1) = inds(train_data(:,1));
train_data(:,2) = inds(train_data(:,2));
train_data(:,3) = inds(train_data(:,3));
% save('train_data.mat', 'train_data');
% load('train_data');
end

