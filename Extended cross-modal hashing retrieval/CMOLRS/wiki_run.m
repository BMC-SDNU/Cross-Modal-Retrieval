% clc;
% clear;

%% set parameters
params = struct();
% params.pos = 1; % corresponding
params.pos = 2; % the same class

params.tri_num = 5e5; % number of training triplets

% params.dir = 1; % image to text
params.dir = 2; % text to image

params.rank = 32; % rank in Eqn(7)
params.base_margin = 20; % beta in Eqn(4)
params.alph = 0.5; % alpha in Eqn(4)
params.step_size = 10;

%% load data
load('data/wiki/data_cnnfc7_zj_zmnor.mat');
pca = 1;
if pca == 1
    options_=[];
    options_.PCARatio = 0.95;
    [eigvector,eigvalue] = myPCA(img_tr,options_);
    img_tr = img_tr*eigvector;
    img_te = img_te*eigvector;
    [eigvector,eigvalue] = myPCA(txt_tr,options_);
    txt_tr = txt_tr*eigvector;
    txt_te = txt_te*eigvector;
end
[N_te, di] = size(img_te);
[N_tr, dt] = size(txt_tr);
% generate triplets
data_train = gen_tr_tri_uc(label_tr, params);
label_tr = scale2vec(label_tr); % change the representation of training labels
fprintf('finish generating triplets.\n');

%%
A = randn(di,params.rank);
B = randn(dt,params.rank);
batches = 10;
params.batch_size = params.tri_num / batches;
time_used = zeros(1,batches);
fout = fopen('record_wiki.txt', 'a');
fprintf(fout, '--------------------------------------------------\n');
begin_time = clock;
for i = 1 : batches
    [model,parms] = loreta_similarity_margin(A, B, img_tr',txt_tr', label_tr', (i-1)*params.batch_size, data_train, params);
    A = model.A;
    B = model.B;
    smatrix = img_te * A * B' * txt_te';
    fprintf('[%s] img search txt:\n', datestr(now,31));
    test_s_map(smatrix, label_te, label_te, fout);
    fprintf('[%s] txt search img:\n', datestr(now,31));
    test_s_map(smatrix', label_te, label_te, fout);
    fprintf('iteration %d out of %d, time elapsed %g sec, %d violated \n',i,...
        batches, etime(clock,begin_time), sum(model.loss));
end
fclose(fout);