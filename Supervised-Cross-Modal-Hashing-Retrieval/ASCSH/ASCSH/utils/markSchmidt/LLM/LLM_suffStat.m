function [ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStat(param,samples,nStates,edges2,edges3,edges4,edges5,edges6,edges7)

nSamples = size(samples,1);
nNodes = size(samples,2);
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

ss1 = zeros(nNodes,nStates-1);
switch param
    case {'I','C','S'}
        ss2 = zeros(nEdges2,1);
        ss3 = zeros(nEdges3,1);
        ss4 = zeros(nEdges4,1);
        ss5 = zeros(nEdges5,1);
        ss6 = zeros(nEdges6,1);
        ss7 = zeros(nEdges7,1);
    case 'P'
        ss2 = zeros(nStates,nEdges2);
        ss3 = zeros(nStates,nEdges3);
        ss4 = zeros(nStates,nEdges4);
        ss5 = zeros(nStates,nEdges5);
        ss6 = zeros(nStates,nEdges6);
        ss7 = zeros(nStates,nEdges7);
    case 'F'
        ss2 = zeros(nStates,nStates,nEdges2);
        ss3 = zeros(nStates,nStates,nStates,nEdges3);
        ss4 = zeros(nStates,nStates,nStates,nStates,nEdges4);
        ss5 = zeros(nStates,nStates,nStates,nStates,nStates,nEdges5);
        ss6 = zeros(nStates,nStates,nStates,nStates,nStates,nStates,nEdges6);
        ss7 = zeros(nStates,nStates,nStates,nStates,nStates,nStates,nStates,nEdges7);
end

for s = 1:nSamples
    y = samples(s,:);
    for n = 1:nNodes
        if y(n) < nStates
            ss1(n,y(n)) = ss1(n,y(n)) + 1;
        end
    end
    for e = 1:nEdges2
        n1 = edges2(e,1);
        n2 = edges2(e,2);
        switch param
            case 'C'
                if y(n1)==1 && y(n2)==1
                    ss2(e) = ss2(e) + 1;
                end
            case 'I'
                if y(n1)==y(n2)
                    ss2(e) = ss2(e) + 1;
                end
            case 'P'
                if y(n1)==y(n2)
                    ss2(y(n1),e) = ss2(y(n1),e) + 1;
                end
            case 'S'
                if mod(y(n1)+y(n2),2)
                    ss2(e) = ss2(e) + 1;
                end
            case 'F'
                ss2(y(n1),y(n2),e) = ss2(y(n1),y(n2),e) + 1;
        end
    end
    for e = 1:nEdges3
        n1 = edges3(e,1);
        n2 = edges3(e,2);
        n3 = edges3(e,3);
        switch param
            case 'C'
                if y(n1)==1 && y(n2)==1 && y(n3)==1
                    ss3(e) = ss3(e) + 1;
                end
            case 'I'
                if y(n1)==y(n2) && y(n2)==y(n3)
                    ss3(e) = ss3(e) + 1;
                end
            case 'P'
                if y(n1)==y(n2) && y(n2)==y(n3)
                    ss3(y(n1),e) = ss3(y(n1),e) + 1;
                end
            case 'S'
                if mod(y(n1)+y(n2)+y(n3),2)
                    ss3(e) = ss3(e) + 1;
                end
            case 'F'
                ss3(y(n1),y(n2),y(n3),e) = ss3(y(n1),y(n2),y(n3),e) + 1;
        end
    end
    for e = 1:nEdges4
        n1 = edges4(e,1);
        n2 = edges4(e,2);
        n3 = edges4(e,3);
        n4 = edges4(e,4);
        switch param
            case 'C'
                if y(n1)==1 && y(n2)==1 && y(n3)==1 && y(n4)==1
                    ss4(e) = ss4(e) + 1;
                end
            case 'I'
                if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4)
                    ss4(e) = ss4(e) + 1;
                end
            case 'P'
                if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4)
                    ss4(y(n1),e) = ss4(y(n1),e) + 1;
                end
            case 'S'
                if mod(y(n1)+y(n2)+y(n3)+y(n4),2)
                    ss4(e) = ss4(e) + 1;
                end
            case 'F'
                ss4(y(n1),y(n2),y(n3),y(n4),e) = ss4(y(n1),y(n2),y(n3),y(n4),e) + 1;
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
                if y(n1)==1 && y(n2)==1 && y(n3)==1 && y(n4)==1 && y(n5)==1
                    ss5(e) = ss5(e) + 1;
                end
            case 'I'
                if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5)
                    ss5(e) = ss5(e) + 1;
                end
            case 'P'
                if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5)
                    ss5(y(n1),e) = ss5(y(n1),e) + 1;
                end
            case 'S'
                if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5),2)
                    ss5(e) = ss5(e) + 1;
                end
            case 'F'
                ss5(y(n1),y(n2),y(n3),y(n4),y(n5),e) = ss5(y(n1),y(n2),y(n3),y(n4),y(n5),e) + 1;
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
                if y(n1)==1 && y(n2)==1 && y(n3)==1 && y(n4)==1 && y(n5)==1 && y(n6)==1
                    ss6(e) = ss6(e) + 1;
                end
            case 'I'
                if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6)
                    ss6(e) = ss6(e) + 1;
                end
            case 'P'
                if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6)
                    ss6(y(n1),e) = ss6(y(n1),e) + 1;
                end
            case 'S'
                if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5)+y(n6),2)
                    ss6(e) = ss6(e) + 1;
                end
            case 'F'
                ss6(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),e) = ss6(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),e) + 1;
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
                if y(n1)==1 && y(n2)==1 && y(n3)==1 && y(n4)==1 && y(n5)==1 && y(n6)==1 && y(n7) == 1
                    ss7(e) = ss7(e) + 1;
                end
            case 'I'
                if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6) && y(n6)==y(n7)
                    ss7(e) = ss7(e) + 1;
                end
            case 'P'
                if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6) && y(n6)==y(n7)
                    ss7(y(n1),e) = ss7(y(n1),e) + 1;
                end
            case 'S'
                if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5)+y(n6)+y(n7),2)
                    ss7(e) = ss7(e) + 1;
                end
            case 'F'
                ss7(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),y(n7),e) = ss7(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),y(n7),e) + 1;
        end
    end
end
ss1 = ss1/nSamples;
ss2 = ss2/nSamples;
ss3 = ss3/nSamples;
ss4 = ss4/nSamples;
ss5 = ss5/nSamples;
ss6 = ss6/nSamples;
ss7 = ss7/nSamples;
