function [varGroupMatrix] = LLM_makeVarGroupMatrix(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

% Make groups
g1 = zeros(size(w1));
g2 = zeros(size(w2));
for e = 1:nEdges2
    switch param
        case {'I','C','S'}
            g2(e) = e;
        case 'P'
            g2(:,e) = e;
        case 'F'
            g2(:,:,e) = e;
    end
end
g3 = zeros(size(w3));
for e = 1:nEdges3
    switch param
        case {'I','C','S'}
            g3(e) = e;
        case 'P'
            g3(:,e) = e;
        case 'F'
            g3(:,:,:,e) = e;
    end
end
g4 = zeros(size(w4));
for e = 1:nEdges4
    switch param
        case {'I','C','S'}
            g4(e) = e;
        case 'P'
            g4(:,e) = e;
        case 'F'
            g4(:,:,:,:,e) = e;
    end
end
g5 = zeros(size(w5));
for e = 1:nEdges5
    switch param
        case {'I','C','S'}
            g5(e) = e;
        case 'P'
            g5(:,e) = e;
        case 'F'
            g5(:,:,:,:,:,e) = e;
    end
end
g6 = zeros(size(w6));
for e = 1:nEdges6
    switch param
        case {'I','C','S'}
            g6(e) = e;
        case 'P'
            g6(:,e) = e;
        case 'F'
            g6(:,:,:,:,:,:,e) = e;
    end
end
g7 = zeros(size(w7));
for e = 1:nEdges7
    switch param
        case {'I','C','S'}
            g7(e) = e;
        case 'P'
            g7(:,e) = e;
        case 'F'
            g7(:,:,:,:,:,:,:,e) = e;
    end
end

% Make hash that lets you find lower-order edges from higher-order edges
hash = java.util.Hashtable;
for e = 1:nEdges2
    for n = 1:nNodes
        if all(edges2(e,:) ~= n)
            nodes = sort([edges2(e,:) n]);
            key = sprintf('%d %d %d',nodes);
            if hash.containsKey(key)
                hash.put(key,[hash.get(key);e]);
            else
                hash.put(key,e);
            end
        end
    end
end
for e = 1:nEdges3
    for n = 1:nNodes
        if all(edges3(e,:) ~= n)
            nodes = sort([edges3(e,:) n]);
            key = sprintf('%d %d %d %d',nodes);
            if hash.containsKey(key)
                hash.put(key,[hash.get(key);e]);
            else
                hash.put(key,e);
            end
        end
    end
end
for e = 1:nEdges4
    for n = 1:nNodes
        if all(edges4(e,:) ~= n)
            nodes = sort([edges4(e,:) n]);
            key = sprintf('%d %d %d %d %d',nodes);
            if hash.containsKey(key)
                hash.put(key,[hash.get(key);e]);
            else
                hash.put(key,e);
            end
        end
    end
end
for e = 1:nEdges5
    for n = 1:nNodes
        if all(edges5(e,:) ~= n)
            nodes = sort([edges5(e,:) n]);
            key = sprintf('%d %d %d %d %d %d',nodes);
            if hash.containsKey(key)
                hash.put(key,[hash.get(key);e]);
            else
                hash.put(key,e);
            end
        end
    end
end
for e = 1:nEdges6
    for n = 1:nNodes
        if all(edges6(e,:) ~= n)
            nodes = sort([edges6(e,:) n]);
            key = sprintf('%d %d %d %d %d %d %d',nodes);
            if hash.containsKey(key)
                hash.put(key,[hash.get(key);e]);
            else
                hash.put(key,e);
            end
        end
    end
end

ind2 = numel(w1);
ind3 = ind2+numel(w2);
ind4 = ind3+numel(w3);
ind5 = ind4+numel(w4);
ind6 = ind5+numel(w5);
ind7 = ind6+numel(w6);
nVars = ind7+numel(w7);

% Compute number of non-zero elements of varGroupMatrix
switch param
    case {'C','I','S'}
        siz2 = 1;
        siz3 = 1 + 3*siz2;
        siz4 = 1 + 4*siz3;
        siz5 = 1 + 5*siz4;
        siz6 = 1 + 6*siz5;
        siz7 = 1 + 7*siz6;
    case 'P'
        siz2 = nStates;
        siz3 = nStates + 3*siz2;
        siz4 = nStates + 4*siz3;
        siz5 = nStates + 5*siz4;
        siz6 = nStates + 6*siz5;
        siz7 = nStates + 7*siz6;
    case 'F'
        siz2 = nStates^2;
        siz3 = nStates^3 + 3*siz2*nStates;
        siz4 = nStates^4 + 4*siz3*nStates;
        siz5 = nStates^5 + 5*siz4*nStates;
        siz6 = nStates^6 + 6*siz5*nStates;
        siz7 = nStates^7 + 7*siz6*nStates;
end
nGroupInd = nEdges2*siz2 + nEdges3*siz3 + nEdges4*siz4 + nEdges5*siz5 + nEdges6*siz6 + nEdges7*siz7;
varGroupMatrix_i = zeros(nGroupInd,1);
varGroupMatrix_j = zeros(nGroupInd,1);

% Add higher-order edges to lower-order groups
k = 0;
%fprintf('Pairwise\n');
for e = 1:nEdges2
    gNdx = find(g2(:)==e);
    iNdx = ind2+gNdx;
    
    varGroupMatrix_i(k+1:k+length(iNdx)) = iNdx;
    varGroupMatrix_j(k+1:k+length(iNdx)) = e;
    k = k+length(iNdx);
end
super3 = zeros(nEdges3,4);
%fprintf('Threeway\n');
for e = 1:nEdges3
    gNdx = find(g3(:)==e);
    iNdx = ind3+gNdx;
    
    varGroupMatrix_i(k+1:k+length(iNdx)) = iNdx;
    varGroupMatrix_j(k+1:k+length(iNdx)) = nEdges2+e;
    k = k+length(iNdx);
	
	super3(e,1) = nEdges2+e;
	s = 1;
    
    key = sprintf('%d %d %d',edges3(e,:));
    for e2 = hash.get(key)'
        gNdx2 = find(g2(:)==e2);
        [X,Y] = meshgrid(ind3+gNdx,e2);
        
        varGroupMatrix_i(k+1:k+numel(X)) = X(:);
        varGroupMatrix_j(k+1:k+numel(Y)) = Y(:);
        k = k+numel(X);
		
		super3(e,s+1) = e2;
		s = s+1;
    end
end
super4 = zeros(nEdges4,17);
%fprintf('Fourway\n');
for e = 1:nEdges4
    gNdx = find(g4(:)==e);
    iNdx = ind4+gNdx;
    
    varGroupMatrix_i(k+1:k+length(iNdx)) = iNdx;
    varGroupMatrix_j(k+1:k+length(iNdx)) = nEdges2+nEdges3+e;
    k = k+length(iNdx);
	
	super4(e,1) = nEdges2+nEdges3+e;
	s = 1;
    
    key = sprintf('%d %d %d %d',edges4(e,:));
    for e2 = hash.get(key)'
        gNdx2 = find(g3(:)==e2);
        [X,Y] = meshgrid(ind4+gNdx,super3(e2,:));
        varGroupMatrix_i(k+1:k+numel(X)) = X(:);
        varGroupMatrix_j(k+1:k+numel(Y)) = Y(:);
        k = k+numel(X);
		
		super4(e,s+1:s+4) = super3(e2,:);
		s = s+4;
    end
end
super5 = zeros(nEdges5,86);
%fprintf('Fiveway\n');
for e = 1:nEdges5
    gNdx = find(g5(:)==e);
    iNdx = ind5+gNdx;
    varGroupMatrix_i(k+1:k+length(iNdx)) = iNdx;
    varGroupMatrix_j(k+1:k+length(iNdx)) = nEdges2+nEdges3+nEdges4+e;
    k = k+length(iNdx);
	
	super5(e,1) = nEdges2+nEdges3+nEdges4+e;
	s = 1;
    
    key = sprintf('%d %d %d %d %d',edges5(e,:));
    for e2 = hash.get(key)'
        gNdx2 = find(g4(:)==e2);
        [X,Y] = meshgrid(ind5+gNdx,super4(e2,:));
        varGroupMatrix_i(k+1:k+numel(X)) = X(:);
        varGroupMatrix_j(k+1:k+numel(Y)) = Y(:);
        k = k+numel(X);
		
		super5(e,s+1:s+17) = super4(e2,:);
		s = s+17;
    end
end
super6 = zeros(nEdges6,517);
%fprintf('Sixway\n');
for e = 1:nEdges6
    gNdx = find(g6(:)==e);
    iNdx = ind6+gNdx;
    varGroupMatrix_i(k+1:k+length(iNdx)) = iNdx;
    varGroupMatrix_j(k+1:k+length(iNdx)) = nEdges2+nEdges3+nEdges4+nEdges5+e;
    k = k+length(iNdx);
	
	super6(e,1) = nEdges2+nEdges3+nEdges4+nEdges5+e;
	s = 1;
    
    key = sprintf('%d %d %d %d %d %d',edges6(e,:));
    for e2 = hash.get(key)'
        gNdx2 = find(g5(:)==e2);
        [X,Y] = meshgrid(ind6+gNdx,super5(e2,:));
        varGroupMatrix_i(k+1:k+numel(X)) = X(:);
        varGroupMatrix_j(k+1:k+numel(Y)) = Y(:);
        k = k+numel(X);
		
		super6(e,s+1:s+86) = super5(e2,:);
		s = s+86;
    end
end
%fprintf('Sevenway\n');
for e = 1:nEdges7
    gNdx = find(g7(:)==e);
    iNdx = ind7+gNdx;
    varGroupMatrix_i(k+1:k+length(iNdx)) = iNdx;
    varGroupMatrix_j(k+1:k+length(iNdx)) = nEdges2+nEdges3+nEdges4+nEdges5+nEdges6+e;
    k = k+length(iNdx);
    
    key = sprintf('%d %d %d %d %d %d %d',edges7(e,:));
    for e2 = hash.get(key)'
        gNdx2 = find(g6(:)==e2);
        [X,Y] = meshgrid(ind7+gNdx,super6(e2,:));
        varGroupMatrix_i(k+1:k+numel(X)) = X(:);
        varGroupMatrix_j(k+1:k+numel(Y)) = Y(:);
        k = k+numel(X);
    end
end

% Re-order varGroupMatrix to make Dykstra run faster
sizeI = nVars;
sizeJ = nEdges2+nEdges3+nEdges4+nEdges5+nEdges6+nEdges7;
varGroupMatrix = sparse(varGroupMatrix_i(1:k),varGroupMatrix_j(1:k),ones(k,1),sizeI,sizeJ,k);
varGroupMatrix = varGroupMatrix(:,[nEdges2+nEdges3+nEdges4+nEdges5+nEdges6+1:nEdges2+nEdges3+nEdges4+nEdges5+nEdges6+nEdges7 nEdges2+nEdges3+nEdges4+nEdges5+1:nEdges2+nEdges3+nEdges4+nEdges5+nEdges6 nEdges2+nEdges3+nEdges4+1:nEdges2+nEdges3+nEdges4+nEdges5 nEdges2+nEdges3+1:nEdges2+nEdges3+nEdges4 nEdges2+1:nEdges2+nEdges3 1:nEdges2]);
varGroupMatrix = logical(varGroupMatrix);
%imagesc(varGroupMatrix)