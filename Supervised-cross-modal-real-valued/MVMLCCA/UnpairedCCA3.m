function [Wx,D,p_each] = UnpairedCCA3(C_x ,C_z,f_type,f)
%C_x : cell of size 1xN, where N is number of modality. Each cell contains
%the feature matrix in the form nxp. where n is number of data samples in
%that modality and p is number of fetures of that data.

%C_z : cell of size 1xN, where each cell contains label matrix of
%corresponding modality. 

%f_type : determines type of similarity function "dot_product" or "squared_exponent"

%Wx : matrix of eigen vectors , size (sum of features x sum of features)
%D : diagnol matrix of eigen values , size (sum of features x sum of features)
%p_each = matrix of size n_modality x 1 , where each row contains number of
%features in that corresponding modality.
tic;
[~,n_modality] = size(C_x);
if(size(C_x) ~= size(C_z))
    disp("CHECK INPUT");
    return;
    %exit(1);
end
reg = 0.0001;
n_each=zeros(n_modality,1);
p_each=zeros(n_modality,1);

for i = 1:n_modality
    disp(["HERE"]);
    %[coeff,score1] = pca(C_x{1,i});
%     [coeff,score2] = pca(C_x{1,i});
    C_x{1,i} = full(MyNormalization(C_x{1,i}));
    [n_each(i),p_each(i)]=size(C_x{1,i})
end

C_all = zeros(sum(p_each),sum(p_each));
C_diag = zeros(sum(p_each),sum(p_each));
start_point_x  = 1;start_point_y = 1;
%F = zeros(sum(p_each),sum(p_each));

for i = 1:n_modality%
    disp(["i is = ",i]);
    for j = i:n_modality
        disp([i,j]);
        %start_point_y = start_point_y + p_each(j,1);
        %p1 = p_each(i,1);
        %p2 = p_each(j,1);
        %S_ij = zeros(p1,p2);
        F_ij = zeros(n_each(i,1),n_each(j,1));
        n_each_j  = n_each(j,1);%edit
        C_z_i = C_z{ 1 , i };
        C_z_j = C_z{ 1 , j };
        C_x_i = C_x{ 1 , i };
        C_x_j = C_x{ 1 , j };
        s = tic;
        %
        for g = 1:n_each(i,1)
            %s = tic;
            %S_ijg = zeros(p1,p2);
            %disp(g);
            zig = C_z_i(g,:);
            %norm_zig = norm(zig);
            %xig = C_x_i(g,:);
            z_j = C_z_j;
            %x_j = C_x_j;
            for h = 1:n_each_j
                F_ij(g,h) = exp(-1*(norm(zig-z_j(h,:))^2));%f_similarity(zig,z_j( h , :),f_type);
%                 if (label_similarity > 0)
%                     %S_ijg = S_ijg + label_similarity*(xig'*x_j( h , :));
%                     F_ij(g,h) = label_similarity;
%                 end
            end
            %S_ij = S_ij;
            %parsave(sprintf('Sij/S%d%d_%d.mat',i,j,g),S_ijg,g);
            %S_ij = S_ij + S_ijg;
            %S_ij = S_ij + S_ijg;
            %parsave(sprintf('Sij/S%d%d_%d.mat',i,j,g),S_ijg,g);
            %S_ij = x_ij;
            %toc(s)
        end
        %
        %squared root Fij
        %F_ij = sqrtm(F_ij);
        eta1_mat_test = (((C_x_i')*sqrt(F_ij))*C_x_j )/(n_each(i,1)*n_each(j,1));
        
        
        S_ij = eta1_mat_test;
        %S_ij = S_ij/(n_each(i,1)*n_each(j,1));
        
        C_all(start_point_x:start_point_x+p_each(i,1)-1, start_point_y:start_point_y+p_each(j,1)-1) = S_ij;
        if(i~=j)
            S_ji = S_ij';
            C_all(start_point_y:start_point_y+p_each(j,1)-1,start_point_x:start_point_x+p_each(i,1)-1) = S_ji;
        else
            C_diag(start_point_x:start_point_x+p_each(i,1)-1 , start_point_y:start_point_y+p_each(j,1)-1) = S_ij;
        end
        start_point_y = start_point_y+ p_each(j,1);
        toc(s)
    end
    start_point_x = start_point_x+ p_each(i,1);
    start_point_y = start_point_x;
end
size(C_all)
C_all = C_all + reg*eye(sum(p_each),sum(p_each));
C_diag = C_diag + reg*eye(sum(p_each),sum(p_each));
[Wx,D] = eig(double(C_all),double(C_diag));
disp('done eigen decomposition');
execution_time = toc
c = clock;
%save(sprintf('Wx_D_iaprtc_%s_%d_%d.mat',f_type,c(1,4),c(1,5)),'Wx','D');
