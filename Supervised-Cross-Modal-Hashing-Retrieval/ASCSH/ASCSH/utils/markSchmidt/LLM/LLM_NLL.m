function [NLL,g] = LLM_NLL(w,param,nSamples,ss1,ss2,ss3,ss4,ss5,ss6,ss7,edges2,edges3,edges4,edges5,edges6,edges7,useMex)

nNodes = size(ss1,1);
nStates = size(ss1,2)+1;

%% Split Weights
[w1,w2,w3,w4,w5,w6,w7] = LLM_splitWeights(w,param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);

%% Compute Z and marginals
if useMex
    [Z,b1,b2,b3,b4,b5,b6,b7] = LLM_inferC(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
else
    [Z,b1,b2,b3,b4,b5,b6,b7] = LLM_infer(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
end

%% Compute NLL
NLL = 0;
NLL = NLL - sum(w1(:).*ss1(:));
NLL = NLL - sum(w2(:).*ss2(:));
NLL = NLL - sum(w3(:).*ss3(:));
NLL = NLL - sum(w4(:).*ss4(:));
NLL = NLL - sum(w5(:).*ss5(:));
NLL = NLL - sum(w6(:).*ss6(:));
NLL = NLL - sum(w7(:).*ss7(:));
NLL = NLL + log(Z);
NLL = nSamples*NLL;

%% Compute gradient
if nargout > 1
   g = nSamples*[b1(:)-ss1(:) 
       b2(:)-ss2(:)
       b3(:)-ss3(:)
       b4(:)-ss4(:)
       b5(:)-ss5(:)
       b6(:)-ss6(:)
       b7(:)-ss7(:)];
end