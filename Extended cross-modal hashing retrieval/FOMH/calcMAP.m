% compute mean average precision (MAP)
function [MAP, succRate] = calcMAP (orderH, neighbor, topR)
[Q, N] = size(neighbor);
if (~exist('topR', 'var'))
    topR = N;
end
if (size(orderH, 1)~=Q || size(orderH, 2)~=N)
    disp(sprintf('size not match: %d=%d %d=%d', size(orderH, 1), Q, size(orderH, 2), N));
    MAP = 0;
    succRate = 0;
    return;
end
pos = [1: topR];
MAP = 0;
numSucc = 0;
for i = 1: Q
    ngb = neighbor(i, orderH(i, 1:topR));
    nRel = sum(ngb);
    if nRel > 0
        prec = cumsum(ngb) ./ pos;
        ap = mean(prec(ngb));
        MAP = MAP + ap;
        numSucc = numSucc + 1;
    end
end
MAP = MAP / numSucc;
succRate = numSucc / Q;
end
