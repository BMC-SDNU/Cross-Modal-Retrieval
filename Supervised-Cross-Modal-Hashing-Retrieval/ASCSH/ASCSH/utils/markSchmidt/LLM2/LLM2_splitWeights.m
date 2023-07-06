function [w1,w2] = LLM2_splitWeights(w,param,nNodes,nStates,edges)

nEdges = size(edges,1);

w1 = reshape(w(1:(nStates-1)*nNodes),nNodes,nStates-1);
offset = (nStates-1)*nNodes;
switch param
    case {'C','I','S'}
        w2 = w(offset+1:offset+nEdges);
    case 'P'
        w2 = reshape(w(offset+1:offset+nEdges*nStates),nStates,nEdges);
    case 'F'
        w2 = reshape(w(offset+1:offset+nEdges*nStates^2),nStates,nStates,nEdges);
end