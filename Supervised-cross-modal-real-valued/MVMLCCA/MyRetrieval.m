function [mAP,mAP21] = MyRetrieval(Wx,D,X_1_test , X_2_test ,Z_1_test,Z_2_test, ld , D_power)
%ld latent space dimension
normalized_X_1_test = MyNormalization(X_1_test);
normalized_X_2_test = MyNormalization(X_2_test);

[n_1_t,p_1_t] = size(X_1_test);
[n_2_t,p_2_t] = size(X_2_test);

%change%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[a, index] = sort(diag(D),'descend');
D = diag(a);
Wx = Wx(:,index);
W_1 = Wx(1:p_1_t,1:ld);%128 x ld %temp = Wx*(D^D_power);%138*d
W_2 = Wx(p_1_t+1:p_1_t+p_2_t,1:ld);%p_1_t + ld %10 x ld
D_ld = D(1:ld,1:ld);

D_1 = D_ld;
D_2 = D_ld;
%D_2 = D_2(p_1_t+1:p_1_t+ld,p_1_t+1:p_1_t+ld);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% W_1 = Wx(1:p_1_t,1:p_1_t);%128 x ld 
% W_2 = Wx(p_1_t+ 1 :p_1_t +p_2_t, p_1_t+ 1 :p_1_t + p_2_t);
% D_1 = D(1:p_1_t,1:p_1_t);
% D_2 = D(p_1_t+ 1 :p_1_t +p_2_t, p_1_t+ 1 :p_1_t + p_2_t);
% 
% [D_1, index_D_1] = sort(diag(D_1),'descend');
% [D_2, index_D_2] = sort(diag(D_2),'descend');
% 
% D_1 = diag(D_1);
% D_2 = diag(D_2);
% 
% W_1 = W_1(:,index_D_1);
% W_2 = W_2(:,index_D_2);
% 
% W_1 = W_1(1:p_1_t,1:ld);%128 x ld %temp = Wx*(D^D_power);%138*d
% W_2 = W_2(1:p_2_t,1:ld);%p_1_t + ld %10 x ld
% 
% D_1 = D_1(1:ld,1:ld);
% D_2 = D_2(1:ld,1:ld);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D_1 = D_1^D_power;
D_2 = D_2^D_power;

result_matrix = zeros(n_1_t,n_2_t);
for i = 1:n_1_t
    for j = 1 :n_2_t
        result_matrix(i,j) = f_similarity(normalized_X_1_test(i,:)*W_1*D_1,normalized_X_2_test(j,:)*W_2*D_2,"dot_product");
    end
end
[a, index] = sort(result_matrix,2,'descend');
% index(333,1:5)
% index(540,1:5)
% index(36,1:5)
[a21,index21] = sort(result_matrix,'descend');%for reversed testing
% index21(333,1:5)
% index21(540,1:5)
% index21(36,1:5)
% P_1 = normalized_X_1_test*W_1*D_1;
% P_2 = normalized_X_2_test*W_2*D_2;
% result_matrix_euclidian = dist_mat(P_1,P_2);
% [a_euclidian, index_euclidian] = sort(result_matrix,2);


precision_all = zeros(n_1_t,n_2_t);
avg_precision_all = zeros(n_1_t,1);
mAP = 0 ;

precision_all21 = zeros(n_2_t,n_1_t);
avg_precision_all21 = zeros(n_2_t,1);
mAP21 = 0 ;

% precision_all_euclidian = zeros(n_1_t,n_2_t);
% avg_precision_all_euclidian  = zeros(n_1_t,1);
% mAP_euclidian = 0 ;
for i = 1:n_1_t
    temp = 0;
    count = 0;
%     temp_euclidian = 0;
%     count_euclidian = 0;
    for j = 1 :n_2_t
        label_similarity = 0;
        if(f_similarity(Z_1_test(i,:),Z_2_test(index(i,j),:),"dot_product")>0)
            label_similarity =1;
        end
        temp = temp + (label_similarity==1);    
        precision_all(i,j)= temp/j;
        if(label_similarity==1)
            avg_precision_all(i) = avg_precision_all(i) + precision_all(i,j);
            count = count + 1;
        end
        %euclidian
%         label_similarity_euclidian = f_similarity( Z_1_test(i,:),Z_2_test(index_euclidian(i,j),:),"dot_product");
%         temp_euclidian = temp_euclidian + (label_similarity_euclidian == 1);
%         precision_all_euclidian(i,j) = temp_euclidian/j;
%         if(label_similarity_euclidian==1)
%             avg_precision_all_euclidian(i) = avg_precision_all_euclidian(i) + precision_all_euclidian(i,j);
%             count_euclidian = count_euclidian + 1;
%         end
    end
    avg_precision_all(i) = avg_precision_all(i)/count;
%     avg_precision_all_euclidian(i) = avg_precision_all_euclidian(i)/count_euclidian;
end
%testing21
for i = 1:n_2_t
    temp = 0;
    count = 0;
%     temp_euclidian = 0;
%     count_euclidian = 0;
    for j = 1 :n_1_t
        label_similarity = 0;
        if(f_similarity(Z_2_test(i,:),Z_1_test(index21(j,i),:),"dot_product")>0)
            label_similarity =1;
        end
        temp = temp + (label_similarity==1);    
        precision_all21 (i,j)= temp/j;
        if(label_similarity==1)
            avg_precision_all21(i) = avg_precision_all21(i) + precision_all21(i,j);
            count = count + 1;
        end
    end
    avg_precision_all21(i) = avg_precision_all21(i)/count;
end

mAP = sum(avg_precision_all)/n_1_t;
mAP21 = sum(avg_precision_all21)/n_2_t;
%mAP_euclidian = sum(avg_precision_all_euclidian)/n_1_t;

