function [w, best_gamma] = logistic_cv(x, y)
%training logistic classifier with cross-validation

gamma = 2.^(-11:2:3);
Pre = zeros(1,length(gamma));
for i = 1:length(Pre)
    
    pre = 0;
    bs = ceil(length(y)/5);
    for j = 1:5
        ind_test = (j-1)*bs+1:min(j*bs,length(y));
        ind_train = setdiff(1:length(y),ind_test);
        w = logistic(x(ind_train,:),y(ind_train,:),[],gamma(i));
        pred = sigmoid(x(ind_test,:)*w)>0.5;
        pre = pre + nnz(pred==y(ind_test,:))/length(pred);
    end
    pre = pre/5;
    
    Pre(i) = pre;
    
end

[best_pre,ind] = max(Pre);
best_gamma = gamma(ind);
fprintf('best pre=%4.5f, best gamma=%4.5f\n', best_pre, best_gamma);

w = logistic(x,y,[],best_gamma);

end