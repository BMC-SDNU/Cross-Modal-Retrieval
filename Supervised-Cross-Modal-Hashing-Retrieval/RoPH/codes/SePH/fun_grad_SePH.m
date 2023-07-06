function [fun, grad] = fun_grad_SePH(B, P, alpha)
%compute function value and gradients for SePH
%
%input:
% B: initial binary codes
% P: similarity matrix
% alpha: regualization parameter
% 
%output:
% fun: function value
% grad: gradients

N = size(P,1);
B = reshape(B, [], N);

D2 = Euclid2(B, B, 'col', 0);
D2 = 1./(D2/4+1);
Q = D2 - diag(diag(D2));
Q = Q/sum(Q(:));

%function value
tmp = P.*log(P./(Q+eps)+eps);
tmp = tmp - diag(diag(tmp));
fun = sum(sum(tmp)) + alpha*sum(sum((abs(B)-1).^2));

%compute gradients
if(nargout>1)
    R = (P-Q).*D2;
    R = R - diag(diag(R));
    grad = zeros(size(B));
    for i = 1:size(B,2)
        d = repmat(B(:,i),1,N)-B;
        grad(:,i) = d*R(:,i);
    end
    grad = grad + 2*alpha*sign(B).*(abs(B)-1);
    grad = grad(:);
end

end