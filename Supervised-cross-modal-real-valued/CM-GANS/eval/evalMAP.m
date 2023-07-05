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

D = pdist([imgfea_norm; txtfea],'cos');
WIT = -squareform(D);
WIT = WIT(1:te_n_I,te_n_I+1:end);
D = pdist([imgfea; txtfea_norm],'cos');
WTI = -squareform(D);
WTI = WTI(1:te_n_I,te_n_I+1:end);
WTI = WTI';

mapIT = QryonTestBi(WIT, imgcat, txtcat);
disp(['Image->Text: ' num2str(mapIT)]);
mapTI = QryonTestBi(WTI, txtcat, imgcat);
disp(['Text->Image: ' num2str(mapTI)]);