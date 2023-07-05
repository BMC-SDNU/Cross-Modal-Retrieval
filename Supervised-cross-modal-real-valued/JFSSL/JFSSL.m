%Implementation of <Joint Feature Selection and Subspace Learning for
%Cross-Modal Retrieval>
%Email: changsuliao@gmail.com

%Load feature mat.
load 'wikipedia_info/raw_features.mat';
data.train = dlmread('wikipedia_info/trainset_txt_img_cat.list','\t',0,2);
data.test = dlmread('wikipedia_info/testset_txt_img_cat.list','\t',0,2);

fprintf('-----------------------------\n');
fprintf('JFSSL\n');
TRAIN_NUM = 2173;
%Build labels matrix Y(2173x10), build train input matrix X_image, X_text.
[Y, X_image, X_text, L_intra_image, L_intra_text, L_inter] = build_pre_M(data, I_tr, T_tr);
%Initialize W_image and W_text
W_image = eye(128, 10);
W_text = eye(10, 10);
%Convergence in 5 steps.
loop_num = 5;
for num = 1:loop_num
    [W_image, W_text] = update_W_JFSSL(W_image, W_text, X_image, X_text, Y,...
        L_intra_image, L_intra_text, L_inter);
    %Get train MAP results.
    %train_or_test = 0;
    % [MAP_image, MAP_text] = get_MAP_result(data, I_tr, T_tr, W_image, W_text,train_or_test);
    %fprintf('Loop %d train image qurey MAP: %f\n', num, MAP_image);
    %fprintf('Loop %d train text qurey MAP: %f\n', num, MAP_text);
    %Get test MAP results.
    train_or_test = 1;
    [MAP_image, MAP_text] = get_MAP_result(data, I_te, T_te, W_image, W_text,train_or_test);
    fprintf('Loop %d test image qurey MAP: %f\n', num, MAP_image);
    fprintf('Loop %d test qurey MAP: %f\n', num, MAP_text);
end
