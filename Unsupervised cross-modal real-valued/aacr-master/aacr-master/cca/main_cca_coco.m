load(['../dataset/mscoco/log_tfidf_split.mat']);

img_tr = [img_tr;img_de];
txt_tr = [txt_tr;txt_de];
label_tr = [label_tr;label_de];
img_tr = double(img_tr);
img_te = double(img_te);

%%
[Wi_, Wt_] = cca_(img_tr', txt_tr', 0.001);

h = 64;

Wi = Wi_(:, 1 : h);
Wt = Wt_(:, 1 : h);
img_te_proj = img_te * Wi;
txt_te_proj = txt_te * Wt;

%%
fout = fopen('record_coco.txt', 'a');
fprintf(fout, '[%s] --------------------------------------\n', datestr(now,31));
fprintf(fout, '[%s] h=%d \n', datestr(now,31), h);
%% test img2txt
fprintf('img search txt:\n');
fprintf(fout, '[%s] img search txt:\n', datestr(now,31));
% smatrix = img_te_proj * txt_te_proj';
smatrix = 1-pdist2(img_te_proj,txt_te_proj,'cosine');
test_s_map(smatrix, label_te, label_te, fout);
fprintf('txt search img:\n');
fprintf(fout, '[%s] txt search img:\n', datestr(now,31));
test_s_map(smatrix', label_te, label_te, fout);
fprintf(fout, '[%s] --------------------------------------\n', datestr(now,31));
fclose(fout);
% % 
img_tr = img_tr * Wi;
txt_tr = txt_tr * Wt;
img_te = img_te * Wi;
txt_te = txt_te * Wt;
save(['../dataset/mscoco/cca_icptlog_coco_', int2str(h_max),'.mat'], 'img_tr', 'txt_tr', 'label_tr', 'img_te', 'txt_te', 'label_te');