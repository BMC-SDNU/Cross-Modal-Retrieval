function w = auxGroupTraceProject(w,p,groupStart,groupPtr)
for i = 1:length(groupStart)-1
    groupInd = groupPtr(groupStart(i):groupStart(i+1)-1);
    [w(groupInd) w(p+i)] = projectAux(w(groupInd),w(p+i));
end
end

%% Function to solve the projection for a single group
function [w,alpha] = projectAux(w,alpha)
p = length(w);
W = reshape(w,sqrt(p),sqrt(p));
[U,S,V] = svd(W);
[s,alpha] = auxProjectL1(diag(S),alpha);
W = U*setdiag(S,s)*V';
w = W(:);
end
