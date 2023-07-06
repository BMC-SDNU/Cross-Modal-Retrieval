function [X,edges] = LLM2_generate(nSamples,nNodes,nStates,edgeProb,param,useMex)

edges = zeros(0,2);
for n1 = 1:nNodes
    for n2 = 1:nNodes
        if rand < edgeProb
            edges(end+1,:) = [n1 n2];
        end
    end
end
edges = int32(edges);
[w1,w2] = LLM2_initWeights(param,nNodes,nStates,edges);
w1 = randn(size(w1));
w2 = 2*randn(size(w2));
if useMex
    Z = LLM2_inferC(param,w1,w2,edges);
else
    Z = LLM2_infer(param,w1,w2,edges);
end
X = LLM2_sample(param,Z,nSamples,w1,w2,edges);