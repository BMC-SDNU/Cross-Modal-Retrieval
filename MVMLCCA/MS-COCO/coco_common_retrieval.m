function [mAP , mAP21] = coco_common_retrieval(X_1_test,X_2_test,Z_1_test,Z_2_test,k)
% st = tic;
[n_1_t,p_1_t] = size(X_1_test);
[n_2_t,p_2_t] = size(X_2_test);
%{
result_matrix = zeros(n_1_t,n_2_t,'single');
for i = 1:n_1_t
    for j = 1:n_2_t
        result_matrix(i,j) = f_similarity(X_1_test(i,:),X_2_test(j,:),"dot_product");
    end   
end
[~, index_matrix] = sort(result_matrix,2,'descend');
index_matrix = index_matrix(:,k);

[~,index_matrix2] = sort(result_matrix,'descend');
index_matrix2 = index_matrix2(k,:);
result_matrix = 0;%free the large matrix;
%}
%
result_matrix = -1*ones(n_1_t,k);
index_matrix = zeros(n_1_t,k);
result_matrix2 = -1*ones(k,n_2_t);
index_matrix2 = zeros(k,n_2_t);

countk = 0;
for i = 1:n_1_t%100 partitions each
    %ss = tic;
    for j = 1 :n_2_t
        features_similarity =  exp(-1*(norm(X_1_test(i,:)-X_2_test(j,:))^2));%f_similarity(X_1_test(i,:),X_2_test(j,:),"squared_exponent");
        if(features_similarity>result_matrix(i,k))
            %disp("if1")
            %countk=countk+1;
            result_matrix(i,k) = features_similarity;
            [sorted,indexi] = sort(result_matrix(i,:),'descend');
            result_matrix(i,:) = sorted;
            
            index_matrix(i,k) = j ;
            tempi = index_matrix(i,indexi);
            index_matrix(i,:) = tempi;
        end
        
        if(features_similarity>result_matrix2(k,j))
            %countk=countk+1;
            %disp("if2")
            result_matrix2(k,j) = features_similarity;
            [sorted,indexi] = sort(result_matrix2(:,j),'descend');
            result_matrix2(:,j) = sorted;
            
            index_matrix2(k,j) = i ;
            tempi = index_matrix2(indexi,j);
            index_matrix2(:,j) = tempi;
        end
    end
    %toc(ss)
end
%}
% save('result_mat_on_eta1.mat','result_matrix');


% [a, index] = sort(result_matrix,2,'descend');
% [a21,index21] = sort(result_matrix,'descend');%for reversed testing
% P_1 = normalized_X_1_test*W_1*D_1;
% P_2 = normalized_X_2_test*W_2*D_2;
% result_matrix_euclidian = dist_mat(P_1,P_2);
% [a_euclidian, index_euclidian] = sort(result_matrix,2);

% toc(st)

precision_all = zeros(n_1_t,k);
avg_precision_all = zeros(n_1_t,1);
mAP = 0 ;

precision_all21 = zeros(n_2_t,n_1_t);
avg_precision_all21 = zeros(n_2_t,1);
mAP21 = 0 ;

% precision_all_euclidian = zeros(n_1_t,n_2_t);
% avg_precision_all_euclidian  = zeros(n_1_t,1);
% mAP_euclidian = 0 ;
% disp('working on mAP12...');
for i = 1:n_1_t %1000 , 100, 
    temp = 0;
    count = 0;
    %sss = tic;
%     temp_euclidian = 0;
%     count_euclidian = 0;
    for j = 1 :k
        label_similarity = 0;
        if(f_similarity(Z_1_test(i,:),Z_2_test(index_matrix(i,j),:),"dot_product")>0)%(i,1:k)
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
    if(count~=0)
        avg_precision_all(i) = avg_precision_all(i)/count;
    end
%     avg_precision_all_euclidian(i) = avg_precision_all_euclidian(i)/count_euclidian;
    %toc(sss);
end
% disp('working on mAP21...');
%testing21
for i = 1:n_2_t
    temp = 0;
    count = 0;
%     temp_euclidian = 0;
%     count_euclidian = 0;
    for j = 1 :k
        label_similarity = 0;
        if(f_similarity(Z_2_test(i,:),Z_1_test(index_matrix2(j,i),:),"dot_product")>0)
            label_similarity =1;
        end
        temp = temp + (label_similarity==1);    
        precision_all21 (i,j)= temp/j;
        if(label_similarity==1)
            avg_precision_all21(i) = avg_precision_all21(i) + precision_all21(i,j);
            count = count + 1;
        end
    end
    if(count~=0)
        avg_precision_all21(i) = avg_precision_all21(i)/count;
    end
end

mAP = sum(avg_precision_all)/n_1_t;
mAP21 = sum(avg_precision_all21)/n_2_t;
%mAP_euclidian = sum(avg_precision_all_euclidian)/n_1_t;
%toc(st)