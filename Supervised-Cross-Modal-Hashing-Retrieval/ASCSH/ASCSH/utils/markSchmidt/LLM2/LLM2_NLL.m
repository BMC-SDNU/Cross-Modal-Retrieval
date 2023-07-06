function [NLL,g] = LLM_NLL(w,param,nSamples,ss1,ss2,edges,useMex)

nNodes = size(ss1,1);
nStates = size(ss1,2)+1;

%% Split Weights
[w1,w2] = LLM2_splitWeights(w,param,nNodes,nStates,edges);

%% Compute Z and marginals
if useMex
    [Z,b1,b2] = LLM2_inferC(param,w1,w2,edges);
else
    [Z,b1,b2] = LLM2_infer(param,w1,w2,edges);
end

%% Compute NLL
NLL = 0;
NLL = NLL - sum(w1(:).*ss1(:));
NLL = NLL - sum(w2(:).*ss2(:));
NLL = NLL + log(Z);
NLL = nSamples*NLL;

%% Compute gradient
if nargout > 1
   g = nSamples*[b1(:)-ss1(:) 
       b2(:)-ss2(:)];
end