function [rev] = revMatrixAdd(rev,adj,anc,i,j)
nNodes = length(adj);
[p c] = find(adj);
for edge = 1:length(p)
    if rev(p(edge),c(edge)) ~= 1
        if p(edge) == i || c(edge) == j || anc(p(edge),i) || anc(j,c(edge))
            rev(p(edge),c(edge)) = testReversal(anc,p(edge),c(edge));
        end
    end
end
