function [x] = projectNDgroup_Dykstra(x,groupStart,groupPtr,normType)

nGroups = length(groupStart)-1;
nVars = length(x)-nGroups;

maxIter = 1000;
nVarGroups = length(groupPtr);
I = zeros(nVarGroups+nGroups,1);

for i = 1:maxIter
    x_prev = x;
    for g = 1:nGroups
        groupNdx = groupPtr(groupStart(g):groupStart(g+1)-1);
        x_old = x(groupNdx);

        if normType == 2
            [x(groupNdx),x(nVars+g)] = projectL2(x(groupNdx)-I(groupStart(g):groupStart(g+1)-1),x(nVars+g)-I(nVarGroups+g));
        else
            [x(groupNdx),x(nVars+g)] = projectLinf(x(groupNdx)-I(groupStart(g):groupStart(g+1)-1),x(nVars+g)-I(nVarGroups+g));
        end
        I(groupStart(g):groupStart(g+1)-1) = x(groupNdx) - (x_old - I(groupStart(g):groupStart(g+1)-1));
        I(nVarGroups+g) = x(nVars+g) - (x_prev(nVars+g) - I(nVarGroups+g));
    end

   %fprintf('Iter = %d, Res = %.10f\n',i,sum(abs(x-x_prev)));
    if sum(abs(x-x_prev)) < 1e-10
        break;
    end
end
%sum(abs(x-x_prev))
%fprintf('nIters = %d\n',i);
%pause;

end

%% Function to solve the L2 projection for a single group
function [w,alpha] = projectL2(w,alpha)
p = length(w);
nw = norm(w);
    if nw > alpha
       avg = (nw+alpha)/2;
       if avg < 0
           w(:) = 0;
           alpha = 0;
       else
           w = w*avg/nw;
           alpha = avg;
       end 
    end
end

%% Function to solve the Linf projection for a single group
function [w,alpha] = projectLinf(w,alpha)
if ~all(abs(w) <= alpha)
    sorted = [sort(abs(w),'descend');0];
    s = 0;
    for k = 1:length(sorted)

        % Compute Projection with k largest elements
        s = s + sorted(k);
        projPoint = (s+alpha)/(k+1);

        if projPoint > 0 && projPoint > sorted(k+1)
            w(abs(w) >= sorted(k)) = sign(w(abs(w) >= sorted(k)))*projPoint;
            alpha = projPoint;
            break;
        end

        if k == length(sorted)
            % alpha is too negative, optimal answer is 0
            w = zeros(size(w));
            alpha = 0;
        end
    end
end
end