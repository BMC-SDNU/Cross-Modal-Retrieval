function result = norm21(P)
result = 0;
for i = 1:size(P,1)
    result = result + norm(P(i,:));
end