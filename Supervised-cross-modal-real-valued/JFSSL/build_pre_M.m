function [Y, X_image, X_text, L_intra_image, L_intra_text, L_inter] = build_pre_M(data, I_tr, T_tr)
%Build label matrix Y and input matrixes X_image and X_text.
%Return Y, X_image, X_text.
TRAIN_NUM = 2173;
%10 classes.
Y = zeros(TRAIN_NUM, 10);
%Record numbers of each class.
class_num = zeros(1, 10);
class_samples_image = zeros(10, TRAIN_NUM, 128);
class_samples_text = zeros(10, TRAIN_NUM, 10);
for num = 1:TRAIN_NUM
    switch data.train(num, 1)
        case 1
            class_num(1,1) = class_num(1,1) + 1;
            class_samples_image(1, class_num(1,1), 1:128) = I_tr(num, 1:128);
            class_samples_text(1, class_num(1,1), 1:10) = T_tr(num, 1:10);
        case 2
            class_num(1,2) = class_num(1,2) + 1;
            class_samples_image(2, class_num(1,2), 1:128) = I_tr(num, 1:128);
            class_samples_text(2, class_num(1,2), 1:10) = T_tr(num, 1:10);
        case 3
            class_num(1,3) = class_num(1,3) + 1;
            class_samples_image(3, class_num(1,3), 1:128) = I_tr(num, 1:128);
            class_samples_text(3, class_num(1,3), 1:10) = T_tr(num, 1:10);
        case 4
            class_num(1,4) = class_num(1,4) + 1;
            class_samples_image(4, class_num(1,4), 1:128) = I_tr(num, 1:128);
            class_samples_text(4, class_num(1,4), 1:10) = T_tr(num, 1:10);
        case 5
            class_num(1,5) = class_num(1,5) + 1;
            class_samples_image(5, class_num(1,5), 1:128) = I_tr(num, 1:128);
            class_samples_text(5, class_num(1,5), 1:10) = T_tr(num, 1:10);
        case 6
            class_num(1,6) = class_num(1,6) + 1;
            class_samples_image(6, class_num(1,6), 1:128) = I_tr(num, 1:128);
            class_samples_text(6, class_num(1,6), 1:10) = T_tr(num, 1:10);
        case 7
            class_num(1,7) = class_num(1,7) + 1;
            class_samples_image(7, class_num(1,7), 1:128) = I_tr(num, 1:128);
            class_samples_text(7, class_num(1,7), 1:10) = T_tr(num, 1:10);
        case 8
            class_num(1,8) = class_num(1,8) + 1;
            class_samples_image(8, class_num(1,8), 1:128) = I_tr(num, 1:128);
            class_samples_text(8, class_num(1,8), 1:10) = T_tr(num, 1:10);
        case 9
            class_num(1,9) = class_num(1,9) + 1;
            class_samples_image(9, class_num(1,9), 1:128) = I_tr(num, 1:128);
            class_samples_text(9, class_num(1,9), 1:10) = T_tr(num, 1:10);
        case 10
            class_num(1,10) = class_num(1,10) + 1;
            class_samples_image(10, class_num(1,10), 1:128) = I_tr(num, 1:128);
            class_samples_text(10, class_num(1,10), 1:10) = T_tr(num, 1:10);
        otherwise
            fprintf('Class counting error!');
    end
end

%Build intr and inter matrix W_intra_inter.
W_intra_image = zeros(TRAIN_NUM, TRAIN_NUM);
W_intra_text = zeros(TRAIN_NUM, TRAIN_NUM);
W_inter = zeros(TRAIN_NUM, TRAIN_NUM);
D_intra_image = zeros(TRAIN_NUM, TRAIN_NUM);
D_intra_text = zeros(TRAIN_NUM, TRAIN_NUM);
D_inter = zeros(TRAIN_NUM, TRAIN_NUM);


%Build label matrix Y.
%Build X_image and X_text.
X_image = zeros(128, TRAIN_NUM);
X_text = zeros(10, TRAIN_NUM);
Y(1:class_num(1,1), 1) = 1;
temp_image = zeros(class_num(1,1), 128);
temp_image(1:class_num(1,1), 1:128) = class_samples_image(1, 1:class_num(1,1), 1:128);
X_image(1:128, 1:class_num(1,1)) = temp_image';
temp_text = zeros(class_num(1,1), 10);
temp_text(1:class_num(1,1), 1:10) = class_samples_text(1, 1:class_num(1,1), 1:10);
X_text(1:10, 1:class_num(1,1)) = temp_text';

W_inter(1:class_num(1,1), 1:class_num(1,1)) = 1;

for j = 2:10
    tsum = 0;
    for t = 1:j - 1
        tsum = tsum + class_num(1, t);
    end
    Y(tsum + 1: tsum + class_num(1,j),j) = 1;
    
    temp_image = zeros(class_num(1,j), 128);
    temp_image(1:class_num(1,j), 1:128) = class_samples_image(j, 1:class_num(1,j), 1:128);
    X_image(1:128, tsum + 1: tsum + class_num(1,j)) = temp_image';
    
    temp_text = zeros(class_num(1,j), 10);
    temp_text(1:class_num(1,j), 1:10) = class_samples_text(j, 1:class_num(1,j), 1:10);
    X_text(1:10, tsum + 1: tsum + class_num(1,j)) = temp_text';
    
    W_inter(tsum + 1: tsum + class_num(1,j), tsum + 1: tsum + class_num(1,j)) = 1;
end
%Build intra similarity matrix.
beta = 1;%Not much influence
delta = 0.1;%Can produce much influence with different value.
temp_M_image = zeros(TRAIN_NUM, TRAIN_NUM);
temp_M_text = zeros(TRAIN_NUM, TRAIN_NUM);
for i = 1:TRAIN_NUM
    for j = 1:TRAIN_NUM
        temp_M_image(i, j) = sum((X_image(:, i) - X_image(:, j)).^2);
        temp_M_text(i, j) = sum((X_text(:, i) - X_text(:, j)).^2);
    end
end

[~, index_image] = sort(temp_M_image, 2);
[~, index_text] = sort(temp_M_text, 2);
k_num = TRAIN_NUM;
for i = 1:TRAIN_NUM
    for j = 1:k_num%Get the k_num nearest samples.
        W_intra_image(i, index_image(i, j)) = exp(-temp_M_image(i, index_image(i, j)) ./ (2 .* (delta .^ 2)));
        W_intra_text(i, index_text(i, j)) = exp(-temp_M_text(i, index_text(i, j)) ./ (2 .* (delta .^ 2)));
    end
end
%Build degree matrix.
for i = 1:TRAIN_NUM
    D_inter(i, i) = sum(W_inter(i, :));
end
for i = 1:TRAIN_NUM
    for j = 1:TRAIN_NUM
        if W_intra_image(i, j) > 0
            D_intra_image(i, i) = D_intra_image(i, i) + 1;
        end
        if W_intra_text(i, j) > 0
            D_intra_text(i, i) = D_intra_text(i, i) + 1;
        end
    end
end
%Build laplance matrix.
L_intra_image = D_intra_image - beta .* W_intra_image;
L_intra_text = D_intra_text - beta .* W_intra_text;
L_inter = D_inter - W_inter;
