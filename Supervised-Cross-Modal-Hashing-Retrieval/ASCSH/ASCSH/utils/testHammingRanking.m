function testHammingRanking()
qB = [
     1 -1  1 -1;
    -1  1  1 -1;
    ];
rB = [
     1 -1  1  1;
     1 -1 -1 -1;
    -1  1  1  1;
    -1 -1  1  1;
    -1 -1  1 -1;
    -1 -1  1  1;
    ];


queryLabel = [
    1 0 0 1;
    0 1 0 0;
    ];


retrievalLabel = [
    1 1 0 1;
    0 1 0 0;
    0 1 0 1;
    1 1 1 0;
    0 0 0 1;
    1 0 0 0;
    ];


qB = compactbit(qB > 0);
rB = compactbit(rB > 0);

topk = 2: 4;

result = calcMapTopkMapTopkPreTopRecLabel(queryLabel, retrievalLabel, qB, rB, topk);

save result.mat result;
end