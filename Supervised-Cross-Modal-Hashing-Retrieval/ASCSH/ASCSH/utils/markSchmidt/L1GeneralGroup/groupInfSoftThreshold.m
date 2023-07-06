function [w] = groupInfSoftThreshold(w,alpha,lambda,groups)
    nGroups = max(groups);
    for g = 1:nGroups
        wG = w(groups==g);
        wL1 = sign(wG).*projectRandom2C(abs(wG),alpha*lambda(g));
        theta = max(abs(wG-wL1));
        w(groups==g) = sign(wG).*min(abs(wG),theta);
    end
end