function [illegal] = testReversal(anc,i,j)
illegal = 0;
for a = find(anc(:,j))'
    if anc(i,a)==1
        illegal = 1;
        break;
    end
end