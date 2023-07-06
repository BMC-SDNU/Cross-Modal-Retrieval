function [NLL,g] = HLLM2_ugmNLL(w,param,nSamples,ss1,ss2,edgeStruct,inferFunc,varargin)

nNodes = size(ss1,1);
nStates = size(ss1,2)+1;
nEdges = edgeStruct.nEdges;

%% Split Weights
[w1,w2] = LLM2_splitWeights(w,param,nNodes,nStates,edgeStruct.edgeEnds);

%% Form potentials
nodePot = ones(nNodes,nStates);
for n = 1:nNodes
    for s = 1:nStates-1
       nodePot(n,s) = exp(w1(n,s)); 
    end
end
edgePot = ones(nStates,nStates,nEdges);
for e = 1:nEdges
    switch param
        case 'C'
            edgePot(1,1,e) = exp(w2(e));
        case 'I'
            for s = 1:nStates
                edgePot(s,s,e) = exp(w2(e));
            end
        case 'P'
            for s = 1:nStates
                edgePot(s,s,e) = exp(w2(s,e));
            end
        case 'S'
            edgePot(1,2,e) = exp(w2(e));
            edgePot(2,1,e) = exp(w2(e));
        case 'F'
            for s1 = 1:nStates
                for s2 = 1:nStates
                    edgePot(s1,s2,e) = exp(w2(s1,s2,e));
                end
            end
    end
end

%% Compute Z and marginals

[bel1,bel2,logZ] = inferFunc(nodePot,edgePot,edgeStruct,varargin{:});

%% Compute NLL
NLL = 0;
NLL = NLL - sum(w1(:).*ss1(:));
NLL = NLL - sum(w2(:).*ss2(:));
NLL = NLL + logZ;
NLL = nSamples*NLL;

%% Get relevant marginals
b1 = bel1(:,1:end-1)';

b2 = zeros(size(ss2));
for e = 1:nEdges
    switch param
        case 'C'
            b2(e) = bel2(1,1,e);
        case 'I'
            b2(e) = sum(diag(bel2(:,:,e)));
        case 'P'
            for s = 1:nStates
                b2(s,e) = bel2(s,s,e);
            end
        case 'S'
            for s = 1:nStates
                b2(e) = bel2(1,2,e)+bel2(2,1,e);
            end
        case 'F'
            b2(:,:,e) = bel2(:,:,e);
    end
end

%% Compute gradient
if nargout > 1
   g = nSamples*[b1(:)-ss1(:) 
       b2(:)-ss2(:)];
end