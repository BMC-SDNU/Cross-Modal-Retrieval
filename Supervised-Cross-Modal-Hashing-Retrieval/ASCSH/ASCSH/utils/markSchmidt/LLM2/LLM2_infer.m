function [Z,b1,b2] = LLM2_infer(param,w1,w2,edges)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges = size(edges,1);

b1 = zeros(size(w1));
b2 = zeros(size(w2));

y = ones(nNodes,1);
Z = 0;
while 1
    logPot = LLM2_logPot(param,y,w1,w2,edges);
    pot = exp(logPot);
    
    Z = Z+pot;
    
    if nargout > 1
        for n = 1:nNodes
            if y(n) < nStates
                b1(n,y(n)) = b1(n,y(n)) + pot;
            end
        end
        for e = 1:nEdges
            n1 = edges(e,1);
            n2 = edges(e,2);
            switch param
                case 'C'
                    if y(n1) == 1 && y(n2) == 1
                        b2(e) = b2(e) + pot;
                    end
                case 'I'
                    if y(n1)==y(n2)
                        b2(e) = b2(e) + pot;
                    end
                case 'P'
                    if y(n1)==y(n2)
                        b2(y(n1),e) = b2(y(n1),e) + pot;
                    end
                case 'S'
                    if mod(y(n1)+y(n2),2)
                        b2(e) = b2(e) + pot;
                    end
                case 'F'
                    b2(y(n1),y(n2),e) = b2(y(n1),y(n2),e) + pot;
            end
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

if nargout > 1
    b1 = b1./Z;
    b2 = b2./Z;
end

