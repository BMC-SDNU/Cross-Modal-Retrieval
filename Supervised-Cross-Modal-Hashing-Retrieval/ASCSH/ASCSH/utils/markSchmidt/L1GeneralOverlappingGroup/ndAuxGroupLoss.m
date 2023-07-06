function [f,g,H] = ndAuxGroupLoss(w,varGroupMatrix,lambda,funObj)
[p,nGroups] = size(varGroupMatrix);

if nargout == 3
    [f,g,H] = funObj(w(1:p));
else
    [f,g] = funObj(w(1:p));
end

f = f + sum(lambda.*w(p+1:end));
g = [g;lambda.*ones(nGroups,1)];

if nargout == 3
   H = [H zeros(p,nGroups);zeros(nGroups,p) zeros(nGroups,nGroups)]; 
end