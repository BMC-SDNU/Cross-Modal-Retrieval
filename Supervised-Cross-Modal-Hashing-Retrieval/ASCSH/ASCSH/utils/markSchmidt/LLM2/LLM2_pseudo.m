function [pseudoNLL,g] = LLM_pseudo(w,param,Y,Yr,nStates,edges,useMex)

[nInstances,nNodes] = size(Y);
nEdges = size(edges,1);

%% Split Weights
[w1,w2] = LLM2_splitWeights(w,param,nNodes,nStates,edges);
    
pseudoNLL = 0;
g1 = zeros(size(w1));
g2 = zeros(size(w2));

%% Compute pseudo-likelihood

if useMex
pseudoNLL = LLM2_pseudoC(param,Y-1,Yr,g1,g2,edges-1,w1,w2);
else
    for i = 1:nInstances
        
        % Compute conditional potential of each node being in each state
        logpot = zeros(nStates,nNodes);
        for n = 1:nNodes
            for s = 1:nStates-1
                logpot(s,n) = logpot(s,n) + w1(n,s);
            end
        end
        for e = 1:nEdges
            n1 = edges(e,1);
            n2 = edges(e,2);
            
            y1 = Y(i,n1);
            y2 = Y(i,n2);
            
            switch param
                case 'C'
                    if y2 == 1
                        logpot(1,n1) = logpot(1,n1) + w2(e);
                    end
                    if y1 == 1
                        logpot(1,n2) = logpot(1,n2) + w2(e);
                    end
                case 'I'
                    logpot(y2,n1) = logpot(y2,n1) + w2(e);
                    logpot(y1,n2) = logpot(y1,n2) + w2(e);
                case 'P'
                    logpot(y2,n1) = logpot(y2,n1) + w2(y2,e);
                    logpot(y1,n2) = logpot(y1,n2) + w2(y1,e);
                case 'S'
                    logpot(mod(y2,2)+1,n1) = logpot(mod(y2,2)+1,n1) + w2(e);
                    logpot(mod(y1,2)+1,n2) = logpot(mod(y1,2)+1,n2) + w2(e);
                case 'F'
                    for s = 1:nStates
                        logpot(s,n1) = logpot(s,n1) + w2(s,y2,e);
                        logpot(s,n2) = logpot(s,n2) + w2(y1,s,e);
                    end
            end
        end
        
        % Compute conditional normalizing constant and update objective
        logZ = mylogsumexp(logpot');
        for n = 1:nNodes
            pseudoNLL = pseudoNLL - Yr(i)*logpot(Y(i,n),n) + Yr(i)*logZ(n);
        end
        
        % Update gradient
        nodeBel = exp(logpot - repmat(logZ',[nStates 1]));
        for n = 1:nNodes
            y1 = Y(i,n);
            if y1 < nStates
                g1(n,y1) = g1(n,y1) - Yr(i)*1;
            end
            g1(n,:) = g1(n,:) + Yr(i)*nodeBel(1:end-1,n)';
        end
        for e = 1:nEdges
            n1 = edges(e,1);
            n2 = edges(e,2);
            
            y1 = Y(i,n1);
            y2 = Y(i,n2);
            
            switch param
                case 'C'
                    if y1==1 && y2 == 1
                        g2(e) = g2(e) - Yr(i)*2;
                    end
                    if y2 == 1
                        g2(e) = g2(e) + Yr(i)*nodeBel(1,n1);
                    end
                    if y1 == 1
                        g2(e) = g2(e) + Yr(i)*nodeBel(1,n2);
                    end
                case 'I'
                    if y1==y2
                        g2(e) = g2(e) - Yr(i)*2;
                    end
                    g2(e) = g2(e) + Yr(i)*nodeBel(y2,n1);
                    g2(e) = g2(e) + Yr(i)*nodeBel(y1,n2);
                case 'P'
                    if y1==y2
                        g2(y1,e) = g2(y1,e) - Yr(i)*2;
                    end
                    g2(y2,e) = g2(y2,e) + Yr(i)*nodeBel(y2,n1);
                    g2(y1,e) = g2(y1,e) + Yr(i)*nodeBel(y1,n2);
                case 'S'
                    if mod(y1+y2,2)
                        g2(e) = g2(e) - Yr(i)*2;
                    end
                    g2(e) = g2(e) + Yr(i)*nodeBel(mod(y2,2)+1,n1);
                    g2(e) = g2(e) + Yr(i)*nodeBel(mod(y1,2)+1,n2);
                case 'F'
                    g2(y1,y2,e) = g2(y1,y2,e) - Yr(i)*2;
                    for s = 1:nStates
                        g2(s,y2,e) = g2(s,y2,e) + Yr(i)*nodeBel(s,n1);
                        g2(y1,s,e) = g2(y1,s,e) + Yr(i)*nodeBel(s,n2);
                    end
            end
        end
    end
end
g = [g1(:);g2(:)];