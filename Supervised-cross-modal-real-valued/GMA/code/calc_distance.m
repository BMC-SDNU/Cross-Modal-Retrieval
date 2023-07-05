function DIST = calc_distance(X,Y,METRIC)

[x1,foox]=size(X);
[y1,fooy]=size(Y);
assert(foox==fooy, 'vectors have different dimensions');
DIST=zeros(x1,y1);


switch upper(METRIC)
  % Euclidean distance
  case {'L1'}
	for i=1:x1,
		for j=1:y1,
			DIST(i,j)=norm( X(i,:) - Y(j,:), 1);
		end
	end

  % Euclidean distance
  case {'L2','EUCL','EUCLID','EUCLIDEAN'}
	for i=1:x1,
		%disp(i)
		for j=1:y1,
			DIST(i,j)=sqrt( sum( (X(i,:) - Y(j,:)).^2 ));
		end
	end

  % KL divergence
  case {'KL'}
	for i=1:x1,
		for j=1:y1,
			DIST(i,j)=KLdiv( X(i,:) , Y(j,:) );
		end
	end
  
  % centered correlation
  case {'CC'}
	for i=1:x1,
		for j=1:y1,
			DIST(i,j)=cent_corr( X(i,:) , Y(j,:) );
		end
	end

  % normalized correlation
  case {'NC'}
	for i=1:x1,
		for j=1:y1,
			DIST(i,j)=norm_corr( X(i,:) , Y(j,:) );
		end
	end

  otherwise
	disp('Unknown metric!')
end
