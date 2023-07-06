function W = fit_mul_logistic_fun(X, Y)
%fit multiple logistic functions
%
%input:
% X: rows correspond samples
% Y: rows correspond 0-1 label vectors
%
%output:
% W: the fitted model parameter

W = zeros(size(X,2),size(Y,2));
for i = 1:size(Y,2)
    if(i<=1)
        [W(:,i), best_gamma] = logistic_cv(X, Y(:,i));
    else
        W(:,i) = logistic(X, Y(:,i), [], best_gamma);
    end
end

end