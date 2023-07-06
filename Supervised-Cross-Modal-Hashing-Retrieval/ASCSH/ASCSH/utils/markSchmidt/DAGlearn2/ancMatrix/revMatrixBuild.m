function [rev] = revMatrixBuild(adj,anc)
nNodes = length(anc);
[i j] = find(adj);
rev = zeros(nNodes);
for edge = 1:length(i)
    rev(i(edge),j(edge)) = testReversal(anc,i(edge),j(edge));
end
