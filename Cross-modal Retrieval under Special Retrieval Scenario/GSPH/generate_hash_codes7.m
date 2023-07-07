function [A,B,f] = generate_hash_codes7(S,m,n,q)

% In this method I tried a new thing - instead of relaxing the following
% said constraints I optimized over the discrete domain. How did I do it ?
% I basically calculated the function value at f(a_il=1) and f(a_il=-1) and
% then set A(i,l) to be that bit for which the function value decreases. I
% do the same for the B(j,l) bit also.


% Now to find A and B - how to do it ? 
% Randomly initialize the matrices A and B 
a = -1; b = 1;
A = (b-a)*rand(m,q) + a;
A = sign(A);
B = (b-a)*rand(n,q) + a;
B = sign(B);

niter = 100;
tolerance = 1e-6;
for t=1:niter    
    
    if t==1
        f_prev = 1e100;
    else
        f_prev = f(t-1);
    end
    
    % Fix B and update A 
    % Do for all rows of A
    R_old = []; R_new = [];
    for i=1:m
        for l=1:q            
            f_old = 0; f_new = 0; 
            for j=1:n; 
                if l==1
                    R_old(j) = A(i,:)*B(j,:).' - q*S(i,j);
                end
                R_new(j) = R_old(j) - A(i,l)*B(j,l) + (-1)*A(i,l)*B(j,l);
                f_old = f_old + R_old(j).^2;
                f_new = f_new + R_new(j).^2;
            end            
            if f_new<f_old
                % update A(i,l)
                A(i,l) = (-1)*A(i,l);
                % update R_old
                R_old = R_new;
            end            
        end
    end
    
    % Fix A and update B 
    % Do for all rows of B
    R_old = []; R_new = [];
    for j=1:n
        for l=1:q            
            f_old = 0; f_new = 0;
            for i=1:m
                if l==1
                    R_old(i) = A(i,:)*B(j,:).' - q*S(i,j);
                end
                R_new(i) = R_old(i) - A(i,l)*B(j,l) + A(i,l)*(-1)*B(j,l);
                f_old = f_old + R_old(i).^2;
                f_new = f_new + R_new(i).^2;
            end            
            if f_new<f_old
                % update B(j,l)
                B(j,l) = (-1)*B(j,l);
                % update R_old
                R_old = R_new;
            end            
        end
    end    
    
    % Compute the function value 
    f(t) = norm(A*B.' - q*S,'fro');
    
    if (f_prev-f(t))/f_prev <= tolerance
        break;
    end
    
    if mod(t,5)==0
    fprintf('The iteration is : %i and f val is : %f \n', t, f(t));
    end
end
A = sign(A); B = sign(B);