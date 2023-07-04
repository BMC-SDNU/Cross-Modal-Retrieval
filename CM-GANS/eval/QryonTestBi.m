function map_val = QryonTestBi( W, testImgCat, testTxtCat)

query = W;
ImgQuery = query;
TxtQuery = query';

[Y,ImgQuery] = sort(ImgQuery,2,'descend');
[Y,TxtQuery] = sort(TxtQuery,2,'descend');

catImgNum = zeros(length(testImgCat),1);
for i = 1:length(testImgCat)
    catImgNum(i) = sum(testTxtCat==testImgCat(i));
end

[map_val,mapICategory] = evaluateMAP_fast_general(ImgQuery,testImgCat,testTxtCat,catImgNum);
