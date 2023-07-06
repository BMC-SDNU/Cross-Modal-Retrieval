function map = mAP(sim_x,L_tr,L_te,mark)
tn = size(sim_x,2);
ap = zeros(tn,1);
R = 100;
for i = 1 : tn
    if mark == 0
        [~,inxx] = sort(sim_x(:,i),'descend');
    elseif mark == 1
        [~,inxx] = sort(sim_x(:,i));
    end
    inxx = inxx(1:R);
    tr_gt = L_tr(inxx);
    ranks = find(tr_gt == L_te(i))';
    
    %compute AP for the query
    if isempty(ranks)
        ap(i) = 0;
    else
        ap(i) = sum((1:length(ranks))./ranks)/length(ranks);
    end
end

map = mean(ap);