function [Xunique,Xreps] = LLM_unique(X)

nSamples = size(X,1);

Xsorted = sortrows(X);
Xunique = unique(Xsorted,'rows');
nUnique = size(Xunique,1);

Xreps = zeros(nUnique,1);
j = 1;
for i = 1:nSamples
    if ~all(Xsorted(i,:) == Xunique(j,:))
        j = j + 1;
    end
    Xreps(j) = Xreps(j) + 1;
end
Xunique = int32(Xunique);