function [ss1,ss2] = LLM_suffStat(param,samples,nStates,edges)

nSamples = size(samples,1);
nNodes = size(samples,2);
nEdges = size(edges,1);

ss1 = zeros(nNodes,nStates-1);
switch param
    case {'I','C','S'}
        ss2 = zeros(nEdges,1);
    case 'P'
        ss2 = zeros(nStates,nEdges);
    case 'F'
        ss2 = zeros(nStates,nStates,nEdges);
end

for s = 1:nSamples
    y = samples(s,:);
    for n = 1:nNodes
        if y(n) < nStates
            ss1(n,y(n)) = ss1(n,y(n)) + 1;
        end
    end
    for e = 1:nEdges
        n1 = edges(e,1);
        n2 = edges(e,2);
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
end
ss1 = ss1/nSamples;
ss2 = ss2/nSamples;