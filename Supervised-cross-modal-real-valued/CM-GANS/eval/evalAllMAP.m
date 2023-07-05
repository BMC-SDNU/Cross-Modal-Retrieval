load('results/img_common_rep_2000.mat');
imgfea=x;
load('results/txt_common_rep_2000.mat');
txtfea=x;

importdata('../data/test_data/test.list');
imgcat=ans.data;
txtcat=ans.data;

te_n_I = 100;
te_n_T = 100;

imgfea_norm = norm_feature(imgfea);
txtfea_norm = norm_feature(txtfea);

D_norm = pdist([imgfea_norm; txtfea_norm],'cosine');
W = -squareform(D_norm);

WIA = W(1:te_n_I,:);
WTA = W(te_n_I+1:end,:);

mapIA = QryonTestBi(WIA, imgcat, [imgcat;txtcat]);
disp(['Image->All: ' num2str(mapIA)]);
mapTA = QryonTestBi(WTA, txtcat, [imgcat;txtcat]);
disp(['Text->All: ' num2str(mapTA)]);