function [Z,b1,b2,b3,b4,b5,b6,b7] = LLM_infer(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

b1 = zeros(size(w1));
b2 = zeros(size(w2));
b3 = zeros(size(w3));
b4 = zeros(size(w4));
b5 = zeros(size(w5));
b6 = zeros(size(w6));
b7 = zeros(size(w7));

y = ones(nNodes,1);
Z = 0;
while 1
    logPot = LLM_logPot(param,y,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
    pot = exp(logPot);
    
    Z = Z+pot;
    
    if nargout > 1
        for n = 1:nNodes
            if y(n) < nStates
                b1(n,y(n)) = b1(n,y(n)) + pot;
            end
        end
        for e = 1:nEdges2
            n1 = edges2(e,1);
            n2 = edges2(e,2);
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
        for e = 1:nEdges3
            n1 = edges3(e,1);
            n2 = edges3(e,2);
            n3 = edges3(e,3);
            switch param
                case 'C'
                    if y(n1) == 1 && y(n2) == 1 && y(n3) == 1
                        b3(e) = b3(e) + pot;
                    end
                case 'I'
                    if y(n1)==y(n2) && y(n2)==y(n3)
                        b3(e) = b3(e) + pot;
                    end
                case 'P'
                    if y(n1)==y(n2) && y(n2)==y(n3)
                        b3(y(n1),e) = b3(y(n1),e) + pot;
                    end
                case 'S'
                    if mod(y(n1)+y(n2)+y(n3),2)
                        b3(e) = b3(e) + pot;
                    end
                case 'F'
                    b3(y(n1),y(n2),y(n3),e) = b3(y(n1),y(n2),y(n3),e) + pot;
            end
        end
        for e = 1:nEdges4
            n1 = edges4(e,1);
            n2 = edges4(e,2);
            n3 = edges4(e,3);
            n4 = edges4(e,4);
            switch param
                case 'C'
                    if y(n1) == 1 && y(n2) == 1 && y(n3) == 1 && y(n4) == 1
                        b4(e) = b4(e) + pot;
                    end
                case 'I'
                    if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4)
                        b4(e) = b4(e) + pot;
                    end
                case 'P'
                    if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4)
                        b4(y(n1),e) = b4(y(n1),e) + pot;
                    end
                case 'S'
                    if mod(y(n1)+y(n2)+y(n3)+y(n4),2)
                        b4(e) = b4(e) + pot;
                    end
                case 'F'
                    b4(y(n1),y(n2),y(n3),y(n4),e) = b4(y(n1),y(n2),y(n3),y(n4),e) + pot;
            end
        end
        for e = 1:nEdges5
            n1 = edges5(e,1);
            n2 = edges5(e,2);
            n3 = edges5(e,3);
            n4 = edges5(e,4);
            n5 = edges5(e,5);
            switch param
                case 'C'
                    if y(n1) == 1 && y(n2) == 1 && y(n3) == 1 && y(n4) == 1 && y(n5) == 1
                        b5(e) = b5(e) + pot;
                    end
                case 'I'
                    if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5)
                        b5(e) = b5(e) + pot;
                    end
                case 'P'
                    if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5)
                        b5(y(n1),e) = b5(y(n1),e) + pot;
                    end
                case 'S'
                    if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5),2)
                        b5(e) = b5(e) + pot;
                    end
                case 'F'
                    b5(y(n1),y(n2),y(n3),y(n4),y(n5),e) = b5(y(n1),y(n2),y(n3),y(n4),y(n5),e) + pot;
            end
        end
        for e = 1:nEdges6
            n1 = edges6(e,1);
            n2 = edges6(e,2);
            n3 = edges6(e,3);
            n4 = edges6(e,4);
            n5 = edges6(e,5);
            n6 = edges6(e,6);
            switch param
                case 'C'
                    if y(n1) == 1 && y(n2) == 1 && y(n3) == 1 && y(n4) == 1 && y(n5) == 1 && y(n6) == 1
                        b6(e) = b6(e) + pot;
                    end
                case 'I'
                    if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6)
                        b6(e) = b6(e) + pot;
                    end
                case 'P'
                    if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6)
                        b6(y(n1),e) = b6(y(n1),e) + pot;
                    end
                case 'S'
                    if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5)+y(n6),2)
                        b6(e) = b6(e) + pot;
                    end
                case 'F'
                    b6(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),e) = b6(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),e) + pot;
            end
        end
        for e = 1:nEdges7
            n1 = edges7(e,1);
            n2 = edges7(e,2);
            n3 = edges7(e,3);
            n4 = edges7(e,4);
            n5 = edges7(e,5);
            n6 = edges7(e,6);
            n7 = edges7(e,7);
            switch param
                case 'C'
                    if y(n1) == 1 && y(n2) == 1 && y(n3) == 1 && y(n4) == 1 && y(n5) == 1 && y(n6) == 1 && y(n7) == 1
                        b7(e) = b7(e) + pot;
                    end
                case 'I'
                    if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6) && y(n6)==y(n7)
                        b7(e) = b7(e) + pot;
                    end
                case 'P'
                    if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6) && y(n6)==y(n7)
                        b7(y(n1),e) = b7(y(n1),e) + pot;
                    end
                case 'S'
                    if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5)+y(n6)+y(n7),2)
                        b7(e) = b7(e) + pot;
                    end
                case 'F'
                    b7(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),y(n7),e) = b7(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),y(n7),e) + pot;
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
    b3 = b3./Z;
    b4 = b4./Z;
    b5 = b5./Z;
    b6 = b6./Z;
    b7 = b7./Z;
end

