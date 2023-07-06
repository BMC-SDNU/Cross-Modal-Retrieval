function [SC] = DAGlearn2_Prune_SC(X,k,ordered,A)

if nargin < 3
    ordered = 0;
end

if nargin < 4
    A = [];
end

[nSamples,nNodes] = size(X);

C = zeros(nNodes);
if isempty(A) % Observational
    for n1 = 1:nNodes
        for n2 = n1+1:nNodes
            % Compute absolute correlation between n1 and n2
            C(n1,n2) = abs(corr(X(:,n1),X(:,n2)));
        end
    end
else % Interventional
    for n1 = 1:nNodes
        for n2 = [1:n1-1 n1+1:nNodes]
            % Compute absolute correlation between n1 and n2, based on
            % samples where we didn't intervene on n2
            ndx = find(A(:,n2)==0);
            C(n1,n2) = abs(corr(X(ndx,n1),X(ndx,n2)));
        end
    end
end

SC = zeros(nNodes);
if ordered
    for n = 1:nNodes
        [sorted,sortedInd] = sort(C(1:n-1,n),'descend');
        SC(sortedInd(1:min(k,n-1)),n) = 1;
    end
else
    if isempty(A)
        C = C + C';
    end
    for n = 1:nNodes
        [sorted,sortedInd] = sort(C(:,n),'descend');
        SC(sortedInd(1:k),n) = 1;
    end
end
end