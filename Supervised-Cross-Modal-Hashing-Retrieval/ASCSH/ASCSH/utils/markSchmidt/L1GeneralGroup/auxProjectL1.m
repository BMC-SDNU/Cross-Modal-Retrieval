function [p,lambda] = auxProjectL1(w,alpha)

if sum(abs(w)) <= alpha
    p = w;
    lambda = alpha;
    return
else
    [c,sortedInd] = sort(abs(w));
    c = [0;c];
    cold = c;
    aold = alpha;
    for i = 1:length(c)
        ctest = max(0,c-c(i)); % Linear
        atest = alpha+c(i);
        if atest > sum(ctest)
            break;
        end
        cold = ctest; % Linear
        aold = atest; % Linear
    end
    
    lambda = aold + (sum(cold)-aold)/(nnz(cold)+1);
    p = sign(w).*max(0,abs(w)-lambda+alpha);
end
