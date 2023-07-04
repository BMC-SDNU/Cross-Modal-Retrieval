function [map, mapCategory] = evaluateMAP_fast_general(queryResult,queryCat,testCat,catNum)

resFlg = testCat(queryResult);

%tic
ap1 = zeros(length(queryCat),1);
% ap = zeros(size(unique(quaryCatDiff),1),1);
% start = quaryCatDiff(1);
% count = 0;
% sum = 0;
% all = 1;
for i = 1:size(queryResult,1)
    query = resFlg(i,:);
    d = find(query==queryCat(i));
    d = (1:catNum(i))./d(1:end);
    ap1(i) = mean(d);
%     if quaryCatDiff(i)~=start
%         ap(all) = sum / count;
%         sum = 0;
%         count = 0;
%         all = all + 1;
%         start = quaryCatDiff(i);
%     end
%     sum = sum + mean(d);
%     count = count + 1;
end
% ap(all) = sum / count;

% fid=fopen('classMAP2.txt','a');
% for i=1:20
%     a = ap1(find(queryCat==i));
%     fprintf(fid,'class %d : %f\n', i, mean(a));
% end
% fprintf(fid,'\n');
% fclose(fid);
map = mean(ap1);
%toc
Category = unique(queryCat);
for i = 1:length(Category)
%     mapCategory(i,1) = mean(ap(queryCat == Category(i)));
      mapCategory(i,1) = 0;
end

end