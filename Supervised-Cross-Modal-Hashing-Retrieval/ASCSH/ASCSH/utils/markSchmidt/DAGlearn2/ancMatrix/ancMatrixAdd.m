function [anc] = ancMatrixAdd(anc,p,c)
% Adds edge (i,j) to ancestor matrix anc

if anc(p,c) == 1
    return
end

anc(p,c) = 1;
for d = find(anc(c,:))
    anc(p,d) = 1;
end
for a = find(anc(:,p))'
    anc(a,c) = 1;
    for d = find(anc(c,:))
        anc(a,d) = 1;
    end
end