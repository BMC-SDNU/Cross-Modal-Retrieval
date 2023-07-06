function [ cb ] = bitCompact( b )
% b: n * c, orignal binary hash code matrix {0, 1}
% cb£ºn * (c/8)£¬compact hash code matrix, uint8
% Borrowed from Spectral Hashing

[nSamples, nbits] = size(b);
nwords = ceil(nbits/8);
cb = zeros([nSamples nwords], 'uint8');

for j = 1:nbits
    w = ceil(j/8);
    cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
end

end

