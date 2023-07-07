function [ train_data ] = gen_tr_list_mc( labels, params )
t = ceil(params.list_size / 2); % positive examples share more labels than negative examples
k = params.list_size - t; % positive examples share labels with input, negative examples don't share labels with input
s_pos = params.list_size; % total number of triplets for a query
train_data = zeros(params.list_num, 1+s_pos+s_pos);
N_tr = size(labels, 1);
for i = 1 : params.list_num
    q_ind = ceil(rand(1)*N_tr);
    train_data(i, 1) = q_ind;
    c1 = 0;
    c2 = 0;
    while c1 < k && c2 < k
        x_ind = ceil(rand(1)*N_tr);
        rel = sum(bitand(labels(q_ind, :), labels(x_ind, :)));
        if rel > 0
            c1 = c1 + 1;
            train_data(i, 1+c1) = x_ind;
        else
            c2 = c2 + 1;
            train_data(i, 1+s_pos+c2) = x_ind;
        end
    end
    if c1 < k
        for j = c1 + 1 : k
            while 1
                x_ind = ceil(rand(1)*N_tr);
                rel = sum(bitand(labels(q_ind, :), labels(x_ind, :)));
                if rel > 0
                    c1 = c1 + 1;
                    train_data(i, 1+c1) = x_ind;
                    break
                end
            end
        end
    else
        for j = c2 + 1 : k
            while 1
                x_ind = ceil(rand(1)*N_tr);
                rel = sum(bitand(labels(q_ind, :), labels(x_ind, :)));
                if rel == 0
                    c2 = c2 + 1;
                    train_data(i, 1+s_pos+c2) = x_ind;
                    break
                end
            end
        end
    end
    for j = 1 : t
        t1_ind = ceil(rand(1)*N_tr);
        t1_rel = sum(bitand(labels(q_ind, :), labels(t1_ind, :)));
        while 1
            t2_ind = ceil(rand(1)*N_tr);
            t2_rel = sum(bitand(labels(q_ind, :), labels(t2_ind, :)));
            if t1_rel ~= t2_rel
                break
            end
        end
        if t1_rel > t2_rel
            train_data(i, 1+k+j) = t1_ind;
            train_data(i, 1+s_pos+k+j) = t2_ind;
        else
            train_data(i, 1+k+j) = t2_ind;
            train_data(i, 1+s_pos+k+j) = t1_ind;
        end
    end
    if mod(i, 10000) == 0
        fprintf('%d\n', i);
    end
end
% save('train_data.mat', 'train_data');
% load('train_data');
end

