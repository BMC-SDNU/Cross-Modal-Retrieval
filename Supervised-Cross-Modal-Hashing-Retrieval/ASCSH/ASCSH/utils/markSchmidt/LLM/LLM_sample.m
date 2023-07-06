function samples = LLM_sample(param,Z,nSamples,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

samples = zeros(nSamples,nNodes);
u = rand(nSamples,1);
z = 0;
y = ones(nNodes,1);
while 1
    logPot = LLM_logPot(param,y,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
    z = z + exp(logPot);
    
    for s = 1:nSamples
        if z/Z > u(s)
            samples(s,:) = y';
            u(s) = inf;
        end
    end
    
    for n = 1:nNodes
        if y(n) < nStates
            y(n) = y(n)+1;
            break;
        else
            y(n) = 1;
        end
    end
    if y(n)==1
        break;
    end
end

