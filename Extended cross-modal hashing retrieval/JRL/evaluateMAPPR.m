function [map,prcurve,mapCategory] = evaluateMAPPR(queryResult, qurCat, resCat)

% *************************************************************************
% *************************************************************************
% Parameters:
% queryResult: the distance matrix between each query and cross-media data for retrieval
%              dimension : m * n
% qurCat: the category list of each query
%              dimension : m * 1
% resCat: the category list of each cross-media data for retrieval
%              dimension : n * 1
% *************************************************************************
% *************************************************************************

[~,queryResult] = sort(queryResult,2,'descend');
AP = zeros(length(qurCat),1);
recall = zeros(length(qurCat),length(resCat));
precision = zeros(length(qurCat),length(resCat));

for i = 1:length(qurCat)
    resultList = queryResult(i,:);
    relCount = 0 ;
    
    relAll = sum(resCat(resultList) == qurCat(i));
    
    
    for j = 1:length(resCat)
        if resCat(resultList(j)) == qurCat(i)
            relCount = relCount + 1;
            AP(i) = AP(i) + relCount/j;
            precision(i,j) = relCount/j;
            recall(i,j) = relCount/relAll;
            if recall(i,j) > 1
                error('recall > 1!');
            end
        else
            precision(i,j) = relCount/j;
            recall(i,j) = relCount/relAll;
        end
    end
    AP(i) = AP(i)/relCount;
end
map = mean(AP);

labelVocabulary = unique(resCat);
mapCategory = zeros(length(labelVocabulary),1);
count = 0;
for i = 1:length(labelVocabulary)
    queryList = (qurCat==labelVocabulary(i));
    count = count + sum(queryList);
    apList = AP(queryList);
    mapCategory(i) = mean(apList);
end

recall = [zeros(length(qurCat), 1),recall];
precision = [ones(length(qurCat), 1),precision];
precisionValue = zeros(1000,length(qurCat));
count = 0;
for recallValue = 0.001:0.001:1
    count = count + 1;
    flag = recall<recallValue;
    flagPlace = sum(flag,2);
    for j = 1:length(qurCat)
        precisionValue(count, j) = calPrecision(precision(j, flagPlace(j)), recall(j, flagPlace(j)), precision(j, flagPlace(j)+1), recall(j, flagPlace(j)+1),recallValue);
    end
end
recallValue = (0.001:0.001:1)';
precision = mean(precisionValue,2);
prcurve = [precision,recallValue];

end

function result = calPrecision (y1, x1, y2, x2, x)
result = (y2-y1)*(x-x1)/(x2-x1) + y1;
end