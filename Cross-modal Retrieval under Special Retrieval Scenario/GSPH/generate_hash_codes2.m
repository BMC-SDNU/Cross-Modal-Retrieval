function [A,B,f] = generate_hash_codes2(S,m,n,q,percent)

%***************************************************************
%**** THIS IS A MUCH FASTER IMPLEMENTATION OF MY ALGORITHM USING THE
% COORDINATED DESCENT SEARCH METHOD WITH NEWTON UPDATES****
% THIS METHOD USES TABLE LOOKUPS TO SIGNIFICANTLY IMPROVE THE COMPUTATIONAL
% EFFICIENCY. IN ADDITION THERE IS A CHOICE TO SELECT HOW MANY BITS I WANT
% TO UPDATE IN A PARTICULAR LOOP BASED ON THE PARAMETER 'percent'
%***************************************************************

% Now to find A and B - how to do it ? 
% Randomly initialize the matrices A and B 
% I used the closed form for the solutions to a_il and bjl to get the new
% value of the coefficients. Note that this does not use the newton's
% update step.

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
    R1 = [];
    
    H_a = zeros(q,1);
    for l=1:q
        for j=1:n            
            H_a(l) = H_a(l) + B(j,l).^2;
        end
    end
    
    % Randomly Select 10% of the data to update
    xx = randperm(m); xxx = round(percent*m);
    m_new = xx(1:xxx);
    
    for z=1:xxx
        i = m_new(z);
        
        % create the initial lookup table
        for j=1:n; R1(j) = (A(i,:)*B(j,:).' - q*S(i,j)); end
        
        for l=1:q
            % gradient of a_il
            % hessian of a_il
            g_a_il = 0;
            for j=1:n; g_a_il = g_a_il + R1(j)*B(j,l); end
            
            % distance to move = d 
            temp = A(i,l);
            d = max(-1-A(i,l),min(-(g_a_il/H_a(l)),1-A(i,l)));
            A(i,l) = A(i,l) +d;
            
            % update R1 lookup table
            for j=1:n; R1(j) = R1(j) - temp*B(j,l) + A(i,l)*B(j,l); end
        end
    end
    
    % Fix A and update B 
    % Do for all rows of B
    R2 = [];
    
    H_b = zeros(q,1);
    for l=1:q
        for i=1:m            
            H_b(l) = H_b(l) + A(i,l).^2;
        end
    end
    
    % Randomly Select 10% of the data to update
    xx = randperm(n); xxx = round(percent*n);
    n_new = xx(1:xxx);
    
    for z=1:xxx
        j = n_new(z);
        
        % create the initial lookup table
        for i=1:m; R2(i) = (A(i,:)*B(j,:).' - q*S(i,j)); end
        for l=1:q
            % gradient of b_jl
            % hessian of b_jl
            g_b_jl = 0;            
            for i=1:m; g_b_jl = g_b_jl + R2(i)*A(i,l); end
            
            % distance to move = d 
            temp = B(j,l);
            d = max(-1-B(j,l),min(-(g_b_jl/H_b(l)),1-B(j,l)));
            B(j,l) = B(j,l) +d;
            
            % update R2 lookup table
            for i=1:m; R2(i) = R2(i) - A(i,l)*temp + A(i,l)*B(j,l); end
        end
    end    
    
    % Compute the function value 
    f(t) = norm(q*S-A*B.','fro');
    
    if (f_prev-f(t))/f_prev <= tolerance
        break;
    end
    
    if mod(t,10)==0
        fprintf('The iteration is : %i and f val is : %f \n', t, f(t));
    end
    
end
A = sign(A); B = sign(B);