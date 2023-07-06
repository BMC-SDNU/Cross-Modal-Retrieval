% function x = logistic(a, y, w, ridge, param)
%
% Logistic regression.  Design matrix A, targets Y, optional instance
% weights W, optional ridge term RIDGE, optional parameters object PARAM.
%
% W is a vector with length equal to the number of training examples; RIDGE
% can be either a vector with length equal to the number of regressors, or
% a scalar (the latter being synonymous to a vector with all entries the
% same).
%
% PARAM has fields PARAM.MAXITER (an iteration limit), PARAM.VERBOSE
% (whether to print diagnostic information), PARAM.EPSILON (used to test
% convergence), and PARAM.MAXPRINT (how many regression coefficients to
% print if VERBOSE==1).
%
% Model is 
%
%   E(Y) = 1 ./ (1+exp(-A*X))
%
% Outputs are regression coefficients X.
%
% Copyright 2007 Geoffrey J. Gordon
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or (at 
% your option) any later version. 
%
% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

function x = logistic(a, y, w, ridge, param)

% process parameters

[n, m] = size(a);

if ((nargin < 3) || (isempty(w)))
  w = ones(n, 1);
end

if ((nargin < 4) || (isempty(ridge)))
  ridge = 1e-5;
end

if (nargin < 5)
  param = [];
end

if (length(ridge) == 1)
    ridgemat = speye(m) * ridge;
elseif (length(ridge(:)) == m)
    ridgemat = spdiags(ridge(:), 0, m, m);
else
    error('ridge weight vector should be length 1 or %d', m);
end

if (~isfield(param, 'maxiter'))
  param.maxiter = 200;
end

if (~isfield(param, 'verbose'))
  param.verbose = 0;
end

if (~isfield(param, 'epsilon'))
  param.epsilon = 1e-10;
end

if (~isfield(param, 'maxprint'))
  param.maxprint = 5;
end

% do the regression

x = zeros(m,1);
oldexpy = -ones(size(y));
for iter = 1:param.maxiter

  adjy = a * x;
  expy = 1 ./ (1 + exp(-adjy));
  deriv = expy .* (1-expy); 
  wadjy = w .* (deriv .* adjy + (y-expy));
  weights = spdiags(deriv .* w, 0, n, n);

  x = inv(a' * weights * a + ridgemat) * a' * wadjy;

  if (param.verbose)
    len = min(param.maxprint, length(x));
    fprintf('%3d: [',iter);
    fprintf(' %g', x(1:len));
    if (len < length(x))
      fprintf(' ... ');
    end
    fprintf(' ]\n');
  end

  if (sum(abs(expy-oldexpy)) < n*param.epsilon)
    if (param.verbose)
      fprintf('Converged.\n');
    end
    return;
  end
  
  oldexpy = expy;

end

warning('logistic:notconverged', 'Failed to converge');
