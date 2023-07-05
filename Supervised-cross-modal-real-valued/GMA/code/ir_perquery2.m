function  [query, class] = ir_perquery2(gt_quer, distance_mtx, gt_retr)
% distance_mtx : Distance of query from the rest of the datapoints
%                Each row refers to one query
% ground_truth : A ground truth vector of categories (starting from 1)
%
% per_query is a stucture with following elements:
%   map   : Mean Average Precision, for all the queries
%   apr   : struct with the following two fields:
%      .P : Average P/n for all the queries
%      .R : Average R/n for all the queries
%           use the last two to compute P/R and P/n curves
%   pr    : precision recal, to plot against 0:0.001:1
%   cm    : confusion matrix
%
% per_class is a structure with following elements_
%   map              : Mean Average Precision, average over the classes
%   map_class        : vector of Mean Average Precision, for al classes
%
%   apr              : struct with the followgin two fields
%   apr.P            : Average P/n cuve, averaged over the classes
%   apn.R            : Average R/n curve, averaged over the classes
%                      Use the last two together to build the PR curve
%
%   apr_class        : vector of structs (as many as the classes)
%                      with the following fields:
%            .P      : Average P/n cuve, for one class
%            .R      : Average R/n cuve, for one class
%   apr_class
%   pr               : matrix, each row contains the pr curve for a class,
%                      to be plotted vs 0:0.001:1



%testPoints = size(distance_mtx,1); 
[r,c]=size(distance_mtx);
testPoints=c;
queryPoints=r;

%%cat_num = max(ground_truth);
[r,c]=size(gt_quer);
if r>c,
    [r,c]=size(gt_retr);
    if c>r,
        gt_retr=gt_retr';
    end
else
    gt_quer=gt_quer';
    [r,c]=size(gt_retr);
    if c>r,
        gt_retr=gt_retr';
    end
end;
gt_cat=sort(unique([gt_quer;gt_retr]));
cat_num = length(gt_cat); 
%disp(gt_quer);
%disp(gt_retr);
%disp(cat_num);
%disp(gt_cat);

%cardinality of the categories
cat_card = zeros(1,cat_num);
for i = 1 : cat_num
    cat_card(i) = length(find(gt_retr == gt_cat(i)));
end
%cat_card,
%gt_cat',

ROCarea       = zeros(1,testPoints);
%%MAP = zeros(testPoints, cat_num); %rank accuracy
MAP = zeros(queryPoints, cat_num); %rank accuracy
top_5 = zeros(1,testPoints);
top_10 = zeros(1,testPoints);
top_20 = zeros(1,testPoints);
top_40 = zeros(1,testPoints);
top_60 = zeros(1,testPoints);
top_100 = zeros(1,testPoints);

%confusion amtrix
conf = zeros(cat_num);
%precision and recall
%%P = zeros(testPoints);
%%R = zeros(testPoints);
%%pn = zeros(testPoints, testPoints);
pr = [];
P = zeros(queryPoints,testPoints);
R = zeros(queryPoints,testPoints);
pn = zeros(queryPoints, testPoints);

% R-Precision
rprecision = zeros(1,queryPoints);

%%for itext = 1 : testPoints
for itext = 1 : queryPoints
	dist = distance_mtx(itext,:);
	[foo ind] = sort(dist,'ascend'); %(give index in V)
	%most similar image (take the original index)
	%pick the class to which the query belongs to
	for cls = 1:cat_num
	  classe = gt_cat(cls); %ground_truth(itext);
	  %classes of all quesries, from best to worst match
% 	  ind = ind(1:100);
	  classeT = gt_retr(ind);
	  %make 0-1 GT
	  classeGT  = (classeT == classe);
	  %compute the indexes in the rank
	  ranks = find(classeGT)';
	  %compute AP for the query
	  map = sum((1:length(ranks))./ranks)/length(ranks);
	  %store
      if isfinite(map),
        MAP(itext,cls) = map;
      end
    end
    %% change 1
    %truemap(itext) = MAP(itext, gt_quer(itext));
    idx_cls=find(gt_cat == gt_quer(itext));
    truemap(itext) = MAP( itext, idx_cls );
    %% end of change 1
    classe = gt_quer(itext);
    classeT = gt_retr(ind);
    classeGT  = (classeT == classe)';
    pn(itext,:) = cumsum(classeGT)./[1:testPoints];
    %% change 2
    %rprecision(itext) = pn( itext, cat_card(gt_quer(itext)) );
    if cat_card(idx_cls)>0,
        rprecision(itext) = pn( itext, cat_card(idx_cls) );
    else
        % no element on the retrieval set belonging to the query object's
        % category. I.e. R-Precision = div by 0 (NaN)
        rprecision(itext) = NaN;
    end
end


for cls = 1:cat_num
    %% change 3 (if)
	%size(gt_quer)
	%size(gt_cat(cls))
    if sum(gt_quer == gt_cat(cls)) > 0,
        cm(cls,:) = mean(MAP(gt_quer == gt_cat(cls),:));
    end
end

query.map = mean(truemap);
query.cmmap = cm;
query.cm = cm;
query.pn = pn;
query.querymap = truemap;
query.MAP = MAP;
% R-Precision
query.queryrprecision = rprecision;
query.rprecision = mean(rprecision);

class.map_class = diag(cm);
