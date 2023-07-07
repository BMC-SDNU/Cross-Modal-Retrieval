% sqd = sqdist(X[,Y,w]) Matrix of squared (weighted) Euclidean distances d(X,Y)
%
% If X and Y are matrices of row vectors of the same dimension (but possibly
% different number of rows), then sqd is a symmetric matrix containing the
% squared (weighted) Euclidean distances between every row vector in X and
% every row vector in Y.
% The square weighted Euclidean distance between two vectors x and y is:
%   sqd = \sum_d { w_d (x_d - y_d)² }.
%
% sqdist requires memory storage for around two matrices of NxM.
%
% NOTE: while this way of computing the distances is fast (because it is
% vectorised), it is slightly inaccurate due to cancellation error, in that
% points Xn and Ym closer than sqrt(eps) will have distance zero. Basically,
% when |a-b| < sqrt(eps) (approx. 1.5e-8) then a²+b²-2ab becomes numerically
% zero while (a-b)² is nonzero.
%
% In:
%   X: NxD matrix of N row D-dimensional vectors.
%   Y: MxD matrix of M row D-dimensional vectors. Default: equal to X.
%   w: 1xD vector of real numbers containing the weights (default: ones).
% Out:
%   sqd: NxM matrix of squared (weighted) Euclidean distances. sqd(n,m)
%      is the squared (weighted) Euclidean distance between row vectors
%      X(n,:) and Y(m,:).
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2009 by Miguel A. Carreira-Perpinan

function sqd = sqdist(X,Y,w)

if nargin==1	% Fast version for common case sqdist(X)
  x = sum(X.^2,2); sqd = max(bsxfun(@plus,x,bsxfun(@plus,x',-2*X*X')),0);
  return
end

% ---------- Argument defaults ----------
if ~exist('Y','var') | isempty(Y) Y = X; eqXY = 1; else eqXY=0; end;
% ---------- End of "argument defaults" ----------
  
if exist('w','var') & ~isempty(w)
  h = sqrt(w(:)'); X = bsxfun(@times,X,h);
  if eqXY==1 Y = X; else Y = bsxfun(@times,Y,h); end;
end

% The intervector squared distance is computed as (x-y)² = x²+y²-2xy.
% We ensure that no value is negative (which can happen due to precision loss
% when two vectors are very close).
x = sum(X.^2,2);
if eqXY==1 y = x'; else y = sum(Y.^2,2)'; end;
sqd = max(bsxfun(@plus,x,bsxfun(@plus,y,-2*X*Y')),0);

