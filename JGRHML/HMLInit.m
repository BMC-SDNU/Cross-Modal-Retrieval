function [I_te_red, T_te_red, I_tr_red, T_tr_red, L_cat] = HMLInit(I_tr, I_te, T_tr, T_te, trainCat)
 
img_Dim = size(I_tr, 2);
txt_Dim = size(T_tr, 2);
tr_n = size(I_tr, 1);
te_n = size(I_te, 1);
L_cat = zeros(tr_n + te_n, max(trainCat));
for i = 1:tr_n
    L_cat(i,trainCat(i)) = 1;
end

gamma1 = 10000;
gamma2 = 10;
lamda1 = 10;
lamda2 = 10;

[I_tr_n I_te_n] = znorm(I_tr,I_te);
[T_tr_n T_te_n] = znorm(T_tr,T_te);

Y0 = double((repmat(trainCat,1,tr_n))==(repmat(trainCat,1,tr_n))');
for i = 1:tr_n
    Y0(i,i) = 0;
end
D=sum(Y0,2); D=1./sqrt(D); 
D=sparse(diag(D));
L = eye(tr_n)-D*Y0*D;

[A,S,B] = svd(I_tr_n'*T_tr_n, 0);

P_img = A;
P_txt = B;

for i = 1:2
    I_tr_red = (P_img' * I_tr_n')';
    T_tr_red = (P_txt' * T_tr_n')';
    I_te_red = (P_img' * I_te_n')';
    T_te_red = (P_txt' * T_te_n')';
    img_mult = inv( gamma1 * eye(img_Dim) + lamda1 * I_tr_n'*L*I_tr_n + I_tr_n'*I_tr_n) * (I_tr_n'*T_tr_n);
    txt_mult = inv( gamma2 * eye(txt_Dim) + lamda2 * T_tr_n'*L*T_tr_n + T_tr_n'*T_tr_n) * (T_tr_n'*I_tr_n);
    P_img_next =  1 * img_mult * P_txt;
    P_txt_next =  1 * txt_mult * P_img;
    P_img = P_img_next;
    P_txt = P_txt_next;
end