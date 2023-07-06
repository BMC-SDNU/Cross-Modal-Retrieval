function [SC] = DAGlearn2_Prune_SC(X,k,ordered,A)

if nargin < 3
    ordered = 0;
end

if nargin < 4
    A = [];
end

[nSamples,nNodes] = size(X);

MI = zeros(nNodes);
if isempty(A) % Observational
    for n1 = 1:nNodes
        for n2 = n1+1:nNodes
            % Compute mutual infromation between n1 and n2
            p_n1a = sum(X(:,n1)==1)/nSamples;
            p_n1b = 1-p_n1a;
            p_n2a = sum(X(:,n2)==1)/nSamples;
            p_n2b = 1-p_n2a;
            
            p_n1n2a = sum(X(:,n1)==1 & X(:,n2)==1)/nSamples;
            p_n1n2b = sum(X(:,n1)==1 & X(:,n2)==-1)/nSamples;
            p_n1n2c = sum(X(:,n1)==-1 & X(:,n2)==1)/nSamples;
            p_n1n2d = 1-p_n1n2a-p_n1n2b-p_n1n2c;
            
            MI(n1,n2) = - p_n1a*log0(p_n1a) - p_n1b*log0(p_n1b) - ...
                p_n2a*log0(p_n2a) - p_n2b*log0(p_n2b) + ...
                p_n1n2a*log0(p_n1n2a) + p_n1n2b*log0(p_n1n2b) + ...
                p_n1n2c*log0(p_n1n2c) + p_n1n2d*log0(p_n1n2d);
        end
    end
else % Interventional
    for n1 = 1:nNodes
        for n2 = [1:n1-1 n1+1:nNodes]
            % Compute mutual infromation between n1 and n2, based on
            % samples where we didn't intervene on n2
            
            ndx = find(A(:,n2)==0);
            nSamples = length(ndx);
            
            p_n1a = sum(X(ndx,n1)==1)/nSamples;
            p_n1b = 1-p_n1a;
            p_n2a = sum(X(ndx,n2)==1)/nSamples;
            p_n2b = 1-p_n2a;
            
            p_n1n2a = sum(X(ndx,n1)==1 & X(ndx,n2)==1)/nSamples;
            p_n1n2b = sum(X(ndx,n1)==1 & X(ndx,n2)==-1)/nSamples;
            p_n1n2c = sum(X(ndx,n1)==-1 & X(ndx,n2)==1)/nSamples;
            p_n1n2d = 1-p_n1n2a-p_n1n2b-p_n1n2c;
            
            MI(n1,n2) = - p_n1a*log0(p_n1a) - p_n1b*log0(p_n1b) - ...
                p_n2a*log0(p_n2a) - p_n2b*log0(p_n2b) + ...
                p_n1n2a*log0(p_n1n2a) + p_n1n2b*log0(p_n1n2b) + ...
                p_n1n2c*log0(p_n1n2c) + p_n1n2d*log0(p_n1n2d);
        end
    end
end

SC = zeros(nNodes);
if ordered
    for n = 1:nNodes
        [sorted,sortedInd] = sort(MI(1:n-1,n),'descend');
        SC(sortedInd(1:min(k,n-1)),n) = 1;
    end
else
    if isempty(A)
        MI = MI + MI';
    end
    for n = 1:nNodes
        [sorted,sortedInd] = sort(MI(:,n),'descend');
        SC(sortedInd(1:k),n) = 1;
    end
end
end

function [x] = log0(x)
x(x~=0) = log(x(x~=0));
end