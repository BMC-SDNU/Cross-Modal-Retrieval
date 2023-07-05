clear;

load('indice.mat');
indice=indice+1;

load('../t2i_attention/extracted_feature/img_fea.mat');
imgfea_imglstm=x(indice,:);
imgfea_imglstm=imgfea_imglstm(232:end,:);
load('../t2i_attention/extracted_feature/txt_fea.mat');
txtfea_imglstm=x(indice,:);
txtfea_imglstm=txtfea_imglstm(232:end,:);

load('../i2t_attention/extracted_feature/img_fea.mat');
imgfea_txtlstm=x(indice,:);
imgfea_txtlstm=imgfea_txtlstm(232:end,:);
load('../i2t_attention/extracted_feature/txt_fea.mat');
txtfea_txtlstm=x(indice,:);
txtfea_txtlstm=txtfea_txtlstm(232:end,:);

load('category.mat');
teCat=teCat(indice);
imgcat = teCat(232:end);
txtcat = teCat(232:end);
te_n_I = 462;
te_n_T = 462;

W1 = imgfea_imglstm * txtfea_imglstm';
W2 = imgfea_txtlstm * txtfea_txtlstm';
W1_new = mnorm(W1) .* (W2);
W2_new = mnorm(W2) .* (W1);
W1 = W1_new;
W2 = W2_new;

max = 0;
aa = 0;
for i=0:0.01:1
W = W1*i+W2*(1-i);
WIT=W;
WTI=W';
map = 0;
%Image->Text
 [mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WIT, imgcat, txtcat);
%  disp(['Image Query MAP: ' num2str(mapIT)]);
 map = map+mapIT;
%Text->Image
 [mapIT, prIQ_IT, mapICategory_IT, ap_IT] = QryonTestBi(WTI, txtcat, imgcat);
%  disp(['Image Query MAP: ' num2str(mapIT)]);
 map = map+mapIT;
 map = map / 2;
%  disp(['Avg MAP: ' num2str(map)]);
 
 if map>max
     max=map;
     aa=i;
 end
end

disp(['max avg: ' num2str(max)]);
disp(['weight: ' num2str(aa)]);
