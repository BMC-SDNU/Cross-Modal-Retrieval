function [I_tr_red,I_te_red,T_tr_red,T_te_red] = learn_projection(I_tr, I_te, T_tr, T_te, pair, img_Dim, txt_Dim, tr_n_I, tr_n_T, L_cat_img, L_cat_txt, cat_Num, param, k)
iteration = param.iteration;
gamma = param.gamma;
sigma = param.sigma;
lamda = param.lamda;
miu = param.miu;

[I_tr_n, I_te_n] = znorm(I_tr,I_te);
[T_tr_n, T_te_n] = znorm(T_tr,T_te);

LI = graphCons([I_tr_n;I_te_n], k);
LT = graphCons([T_tr_n;T_te_n], k);
LI(isnan(LI)) = 0.01;
LT(isnan(LT)) = 0.01;

I_tr_map = I_tr_n(pair(:,1),:);
T_tr_map = T_tr_n(pair(:,2),:);

rand('seed', 1);
P_img = rand(img_Dim, cat_Num);
rand('seed', 1);
P_txt = rand(txt_Dim, cat_Num);

lastLoss = 0;
for i = 1:iteration
    lossValue = trace(sigma * P_img' * [I_tr_n;I_te_n]' * LI * [I_tr_n;I_te_n] * P_img) + sigma * trace(P_txt' * [T_tr_n;T_te_n]' * LT * [T_tr_n;T_te_n] * P_txt) + ...
        miu*sum(sum((I_tr_n * P_img - L_cat_img(1:tr_n_I,:)).^2)) + miu*sum(sum((T_tr_n * P_txt - L_cat_txt(1:tr_n_T,:)).^2)) + ...
        lamda * sum(sum((I_tr_map * P_img - T_tr_map * P_txt).^2)) + ...
        gamma*norm21(P_img) + gamma*norm21(P_txt);
    disp(['iteration ' num2str(i) ': ' num2str(lossValue) ', change ratio:' num2str((lastLoss - lossValue) / lastLoss)]);

    I_tr_red = (P_img' * I_tr_n')';
    T_tr_red = (P_txt' * T_tr_n')';
    I_te_red = (P_img' * I_te_n')';
    T_te_red = (P_txt' * T_te_n')';
    
    I_tr = I_tr_red;
    I_te = I_te_red;
    T_tr = T_tr_red;
    T_te = T_te_red;
    
    D1 = zeros(img_Dim, img_Dim);
    D2 = zeros(txt_Dim, txt_Dim);
    for D1_i = 1:img_Dim
        D1(D1_i,D1_i) = 1/(2*norm(P_img(D1_i, :)));
    end
    for D2_i = 1:txt_Dim
        D2(D2_i,D2_i) = 1/(2*norm(P_txt(D2_i, :)));
    end
    img_mult = inv(gamma * D1 + sigma * [I_tr_n;I_te_n]'*LI*[I_tr_n;I_te_n] + lamda * I_tr_map'*I_tr_map + miu*I_tr_n'*I_tr_n);
    txt_mult = inv(gamma * D2 + sigma * [T_tr_n;T_te_n]'*LT*[T_tr_n;T_te_n] + lamda * T_tr_map'*T_tr_map + miu*T_tr_n'*T_tr_n);

    P_img_next =  1 * img_mult * (lamda*I_tr_map'*T_tr_map*P_txt + miu*I_tr_n'*L_cat_img(1:tr_n_I,:));
    P_txt_next =  1 * txt_mult * (lamda*T_tr_map'*I_tr_map*P_img + miu*T_tr_n'*L_cat_txt(1:tr_n_T,:));
    P_img = P_img_next;
    P_txt = P_txt_next;

    if lastLoss ~= 0 && (lastLoss - lossValue) / lastLoss < 0.02
        break;
    end
    lastLoss = lossValue;

end