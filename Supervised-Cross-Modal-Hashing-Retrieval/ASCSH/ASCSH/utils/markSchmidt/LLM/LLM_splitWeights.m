function [w1,w2,w3,w4,w5,w6,w7] = LLM_splitWeights(w,param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7)

nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

w1 = reshape(w(1:(nStates-1)*nNodes),nNodes,nStates-1);
offset = (nStates-1)*nNodes;
switch param
    case {'C','I','S'}
        w2 = w(offset+1:offset+nEdges2);
        offset = offset + nEdges2;
        w3 = w(offset+1:offset+nEdges3);
        offset = offset + nEdges3;
        w4 = w(offset+1:offset+nEdges4);
        offset = offset + nEdges4;
        w5 = w(offset+1:offset+nEdges5);
        offset = offset + nEdges5;
        w6 = w(offset+1:offset+nEdges6);
        offset = offset + nEdges6;
        w7 = w(offset+1:offset+nEdges7);
    case 'P'
        w2 = reshape(w(offset+1:offset+nEdges2*nStates),nStates,nEdges2);
        offset = offset + nEdges2*nStates;
        w3 = reshape(w(offset+1:offset+nEdges3*nStates),nStates,nEdges3);
        offset = offset + nEdges3*nStates;
        w4 = reshape(w(offset+1:offset+nEdges4*nStates),nStates,nEdges4);
        offset = offset + nEdges4*nStates;
        w5 = reshape(w(offset+1:offset+nEdges5*nStates),nStates,nEdges5);
        offset = offset + nEdges5*nStates;
        w6 = reshape(w(offset+1:offset+nEdges6*nStates),nStates,nEdges6);
        offset = offset + nEdges6*nStates;
        w7 = reshape(w(offset+1:offset+nEdges7*nStates),nStates,nEdges7);
    case 'F'
        w2 = reshape(w(offset+1:offset+nEdges2*nStates^2),nStates,nStates,nEdges2);
        offset = offset + nEdges2*nStates^2;
        w3 = reshape(w(offset+1:offset+nEdges3*nStates^3),nStates,nStates,nStates,nEdges3);
        offset = offset + nEdges3*nStates^3;
        w4 = reshape(w(offset+1:offset+nEdges4*nStates^4),nStates,nStates,nStates,nStates,nEdges4);
        offset = offset + nEdges4*nStates^4;
        w5 = reshape(w(offset+1:offset+nEdges5*nStates^5),nStates,nStates,nStates,nStates,nStates,nEdges5);
        offset = offset + nEdges5*nStates^5;
        w6 = reshape(w(offset+1:offset+nEdges6*nStates^6),nStates,nStates,nStates,nStates,nStates,nStates,nEdges6);
        offset = offset + nEdges6*nStates^6;
        w7 = reshape(w(offset+1:offset+nEdges7*nStates^7),nStates,nStates,nStates,nStates,nStates,nStates,nStates,nEdges7);
end