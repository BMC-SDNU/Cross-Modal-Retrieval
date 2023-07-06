function [anc] = ancMatrixBuild(adj)
% Constructs ancestor matrix given adjacency matrix

nNodes = length(adj);
order = topSort(adj);
anc = zeros(nNodes);
for c = order'
   for p = find(adj(:,c))'
       anc(p,c) = 1;
       for a = find(anc(:,p))'
           anc(a,c) = 1;
       end
   end
end