function map = mAP(ids, Lbase, Lquery)

nquery = size(ids, 2);
APx = zeros(nquery, 1);
R = size(Lbase,1); % Configurable

for i = 1 : nquery
    label = Lquery(i, :);
    label(label == 0) = -1;
    idx = ids(:, i);
    imatch = sum(bsxfun(@eq, Lbase(idx(1:R), :), label), 2) > 0;
    LX = sum(imatch);
    Lx = cumsum(imatch);
    Px = Lx ./ (1:R)';
    if LX ~= 0
        APx(i) = sum(Px .* imatch) / LX;
    end
end
map = mean(APx);

end
