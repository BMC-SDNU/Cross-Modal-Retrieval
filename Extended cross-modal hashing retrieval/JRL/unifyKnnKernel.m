function W_dot = unifyKnnKernel(I_tr_red, I_te_red, T_tr_red, T_te_red,tr_n_I, te_n_I, tr_n_T, te_n_T, trImgCat, trTxtCat, k)

D = pdist([I_tr_red; I_te_red; T_tr_red; T_te_red],'euclidean');
Z = squareform(D);

Z = -Z;
Z1 = 1./(1+exp(-Z));

Z = Z1;

W = Z([tr_n_I+1:tr_n_I+te_n_I, tr_n_I+te_n_I+tr_n_T+1:tr_n_I+te_n_I+tr_n_T+te_n_T], [1:tr_n_I, tr_n_I+te_n_I+1:tr_n_I+te_n_I+tr_n_T]);

[KN,I] = sort(W,2,'descend');
for i = 1:te_n_I + te_n_T
    knn = KN(i,:);
    knn = [knn(1:k), zeros(1,tr_n_I + tr_n_T-k)];
    W(i,I(i,:)) = knn;
end
WI = W(1:te_n_I, :);
WT = W(te_n_I+1:te_n_I+te_n_T, :);

WI_s = sum(WI, 2);
WT_s = sum(WT, 2);
WI = WI./repmat(WI_s, 1, tr_n_I+tr_n_T);
WT = WT./repmat(WT_s, 1, tr_n_T+tr_n_I);
W = [WI;WT];

Y0 = double((repmat([trImgCat;trTxtCat],1,(tr_n_I+tr_n_T)))==(repmat([trImgCat;trTxtCat],1,(tr_n_I+tr_n_T)))');

W_dot = W * Y0 * W';
