function [w1,w2] = LLM2_initWeights(param,nNodes,nStates,edges)

nEdges = size(edges,1);

w1 = zeros(nNodes,nStates-1);
switch param
    case {'I','C','S'}
        w2 = zeros(nEdges,1);
    case 'P'
        w2 = zeros(nStates,nEdges);
    case 'F'
        w2 = zeros(nStates,nStates,nEdges);
end