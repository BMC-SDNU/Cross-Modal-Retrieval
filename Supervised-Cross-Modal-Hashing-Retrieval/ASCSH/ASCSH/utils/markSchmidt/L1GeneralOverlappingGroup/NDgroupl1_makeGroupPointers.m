function [groupStart,groupPtr] = NDgroupl1_makeGroupPointers(varGroupMatrix)
% Note: varGroupMatrix can be sparse
[nVars,nGroups] = size(varGroupMatrix);

% Make pointers indicating starts of groups
groupNumbers = sum(varGroupMatrix)';
groupStart = 1+[0;cumsum(groupNumbers(1:end-1));sum(groupNumbers)];

groupPtr = zeros(0,1);
for g = 1:nGroups
    groupNdx = find(varGroupMatrix(:,g));
    groupPtr(end+1:end+length(groupNdx),1) = groupNdx;
end
groupStart = int32(groupStart);
groupPtr = int32(groupPtr);