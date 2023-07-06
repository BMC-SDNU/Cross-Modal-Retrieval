function [f] = groupL1regularizer(w,lambda,groups)
nGroups = max(groups);
f = 0;
for g = 1:nGroups
   f = f + lambda(g)*max(abs(w(groups==g)));
end
end