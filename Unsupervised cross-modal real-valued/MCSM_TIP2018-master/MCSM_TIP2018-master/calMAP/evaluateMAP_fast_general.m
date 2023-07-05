function [map, mapCategory] = evaluateMAP_fast_general(queryResult,queryCat,testCat,catNum)

resFlg = testCat(queryResult);

%tic
ap1 = zeros(length(queryCat),1);
for i = 1:size(queryResult,1)
    query = resFlg(i,:);
    d = find(query==queryCat(i));
    d = (1:catNum(i))./d(1:end);
    ap1(i) = mean(d);
end

map = mean(ap1);
%toc
Category = unique(queryCat);
for i = 1:length(Category)
%     mapCategory(i,1) = mean(ap(queryCat == Category(i)));
      mapCategory(i,1) = 0;
end

end