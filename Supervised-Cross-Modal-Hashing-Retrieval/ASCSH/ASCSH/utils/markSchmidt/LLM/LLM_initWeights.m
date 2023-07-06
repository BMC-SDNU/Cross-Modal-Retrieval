function [w1,w2,w3,w4,w5,w6,w7] = LLM_initWeights(param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7)

nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

w1 = zeros(nNodes,nStates-1);
switch param
    case {'I','C','S'}
        w2 = zeros(nEdges2,1);
        w3 = zeros(nEdges3,1);
        w4 = zeros(nEdges4,1);
        w5 = zeros(nEdges5,1);
        w6 = zeros(nEdges6,1);
        w7 = zeros(nEdges7,1);
    case 'P'
        w2 = zeros(nStates,nEdges2);
        w3 = zeros(nStates,nEdges3);
        w4 = zeros(nStates,nEdges4);
        w5 = zeros(nStates,nEdges5);
        w6 = zeros(nStates,nEdges6);
        w7 = zeros(nStates,nEdges7);
    case 'F'
        w2 = zeros(nStates,nStates,nEdges2);
        w3 = zeros(nStates,nStates,nStates,nEdges3);
        w4 = zeros(nStates,nStates,nStates,nStates,nEdges4);
        w5 = zeros(nStates,nStates,nStates,nStates,nStates,nEdges5);
        w6 = zeros(nStates,nStates,nStates,nStates,nStates,nStates,nEdges6);
        w7 = zeros(nStates,nStates,nStates,nStates,nStates,nStates,nStates,nEdges7);
end