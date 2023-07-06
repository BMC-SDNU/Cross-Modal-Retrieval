function [anc] = ancMatrixDelete(anc,p,c,adj)
% Adds edge (i,j) from ancestor matrix anc

%adj(p,c) = 0;
for ps = find(adj(:,c))'
   if anc(p,ps) == 1
       return
   end
end

order = topSort(adj);
start = find(order==c);
anc(:,order(start:end))=0;
for j = order(start:end)'
    for i = find(adj(:,j))'
        anc(i,j) = 1;
        for a = find(anc(:,i))'
            anc(a,j) = 1;
        end
    end
end