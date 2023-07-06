function samples = LLM_sample(param,Z,nSamples,w1,w2,edges)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges = size(edges,1);

samples = zeros(nSamples,nNodes);
u = rand(nSamples,1);
z = 0;
y = ones(nNodes,1);
while 1
    logPot = LLM2_logPot(param,y,w1,w2,edges);
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

