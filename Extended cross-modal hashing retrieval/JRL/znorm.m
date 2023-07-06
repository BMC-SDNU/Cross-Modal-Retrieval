function [tr_norm te_norm] = znorm(tr,te)

tr_n = size(tr,1);
te_n = size(te,1);
tr_mean = mean(tr,1);
tr_std = std(tr,1);
if min(tr_std) == 0
    tr_std = tr_std + 0.001;
end
tr_norm = (tr-repmat(tr_mean,tr_n,1))./repmat(tr_std,tr_n,1);
te_norm = (te-repmat(tr_mean,te_n,1))./repmat(tr_std,te_n,1);
