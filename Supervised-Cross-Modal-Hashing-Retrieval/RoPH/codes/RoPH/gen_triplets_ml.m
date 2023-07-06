function [triplets, sims] = gen_triplets_ml(label, kpos_neg)
%generate triplets for multi-label data sets
%
%inputs:
% label: training label vectors
% kpos_neg: number of triplets for each sample
%
%outputs:
% triplets: #triplets*3
% sims: indicator of positive/negative triplet

% This code is written by Kun Ding (kding@nlpr.ia.ac.cn).

assert(length(unique(label(:)))==2);%should be binary labels
assert(min(label(:))==0);
assert(max(label(:))==1);

label = double(label);
N = size(label,2);

%generate training list
triplets = cell(N,1);
sims = cell(N,1);
parfor i = 1:N
    if(mod(i,1000)==0)
        fprintf('%d...',i);
    end
    if(mod(i,10000)==0)
        fprintf('\n');
    end
    [triplets{i},sims{i}] = deal_one_sample(label,i,kpos_neg);
end
triplets = cell2mat(triplets);
sims = cell2mat(sims);
fprintf('\n');

end

function [triplets,sims] = deal_one_sample(label,i,kpos_neg)

this_label = label(:,i);%label of i-th training sample
s = full(this_label'*label);
max_s = max(s);
sim = 0:max_s;
[num1,~] = hist(s,sim);
ind_nnz = find(num1~=0);
groups = cell(1,max_s+1);
cpl_groups = cell(1,max_s+1);
num2 = zeros(1,max_s+1);
for g = 1:max_s+1
    groups{g} = find(s==g-1);
    cpl_groups{g} = find(s<g-1);%find(s~=g-1)
    num2(g) = length(cpl_groups{g});
end
ind1 = randsample(ind_nnz, kpos_neg, true, sim(ind_nnz));
[num,cen] = hist(ind1,1:max_s+1);
t = 0;
ind_pos = zeros(kpos_neg,1);
ind_neg = zeros(kpos_neg,1);
for j = 1:length(num)
    if(num(j)~=0)
        ind_pos(t+1:t+num(j)) = randsample(groups{cen(j)},num(j),num1(cen(j))<num(j));
        ind_neg(t+1:t+num(j)) = randsample(cpl_groups{cen(j)},num(j),num2(cen(j))<num(j));
        t = t + num(j);
    end
end
sims = 2*(s(ind_pos)>s(ind_neg))-1;
sims = sims(:);
triplets = [repmat(i,kpos_neg,1),ind_pos,ind_neg];

end