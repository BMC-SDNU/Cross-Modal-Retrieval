
%Code for training on COCO dataset,
I_tr_ResNet101 = full(MyNormalization(I_tr_ResNet101));
T_tr_Word2Vec_caption = full(MyNormalization(T_tr_Word2Vec_caption));
tic;
F_11_1 = zeros(20000,20000);
eta1_11 = zeros(2048,2048);
eta1_12 = zeros(2048,300);
eta1_22 = zeros(300,300);
arr_end = [16000,32000,48000,64000,82783];
arr_sub = [0,16000,32000,48000,64000];
arr_start = [1,16001,32001,48001,64001];
arr_sub_f = [16000,16000,16000,16000,18783];
for pa1 = 1:5
    for pa = 1:5
        st = tic;
        for g =  arr_start(pa1):arr_end(pa1)     
            for h = arr_start(pa):arr_end(pa)
                      %  disp([g,h]);
                        F_11_1(g-arr_sub(pa1),h-arr_sub(pa)) = exp(-1*(norm(Z_tr(g,:)-Z_tr(h,:))^2));%f_similarity(zig,z_j( h , :),f_type);
            end
        end
        eta1_11 = eta1_11+ (I_tr_ResNet101(arr_start(pa1):arr_end(pa1),:)'*F_11_1(1:arr_sub_f(pa1),1:arr_sub_f(pa)))*I_tr_ResNet101(arr_start(pa):arr_end(pa),:);
        eta1_12 = eta1_12 + I_tr_ResNet101(arr_start(pa1):arr_end(pa1),:)'*F_11_1(1:arr_sub_f(pa1),1:arr_sub_f(pa))*T_tr_Word2Vec_caption(arr_start(pa):arr_end(pa),:);
        eta1_22 = eta1_22+ T_tr_Word2Vec_caption(arr_start(pa1):arr_end(pa1),:)'*F_11_1(1:arr_sub_f(pa1),1:arr_sub_f(pa))*T_tr_Word2Vec_caption(arr_start(pa):arr_end(pa),:);
        toc(st)
    end
end
eta1_11 = eta1_11/(82783*82783);
eta1_12 = eta1_12/(82783*82783);
eta1_22 = eta1_22/(82783*82783);

C_all = [eta1_11 eta1_12 ; eta1_12' eta1_22];
C_diag = ones_diag.*C_all;

C_all = C_all + regmat;
C_diag = C_diag + regmat;
[Wx,D] = eig(double(C_all),double(C_diag));
% save('coco_all_eta1_11_12_22_Wx_D.mat','eta1_11','eta1_12','eta1_22','Wx','D');


