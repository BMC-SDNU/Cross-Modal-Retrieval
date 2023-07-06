function map = mAP(sim_x,L_tr,L_te,R)
[~,cat] = size(L_tr);
multiLabel = cat > 1;
%R = size(L_tr,1);
%R=50
if multiLabel 
    Label = L_tr * L_te';
end
tn = size(sim_x,2);
ap = zeros(tn,1);
for i = 1 : tn
    [~,inxx] = sort(sim_x(:,i),'descend');%inxx保存与第i个测试样本hammingDist最小的前R个database样本所在的位置

    
    if multiLabel
       inxx = inxx(1:R);
       %tr_gt = L_tr(inxx);
       %ranks = find(tr_gt == L_te(i))';  
       ranks = find(Label(inxx,i)>0)';
    
    else
       inxx = inxx(1:R);
       tr_gt = L_tr(inxx);
       ranks = find(tr_gt == L_te(i))';
    end 
    %compute AP for the query
    if isempty(ranks)
        ap(i) = 0;
    else
        ap(i) = sum((1:length(ranks))./ranks)/length(ranks);
    end
end

map = mean(ap);

