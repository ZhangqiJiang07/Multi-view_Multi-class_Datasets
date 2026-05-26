function [Z]= partialanchors(X,index,m,k)
% Construct the anchor graphs for each view
V = max(size(X));
for v=1:V
    t = find(index(:,v)==1);
    % Z{v} = myconstructAnchor(X{v}(t,:), m(v), k);
    T = X{v}(t,:)';
    if m(v) > size(T, 2)
        m(v) = size(T, 2)-3;
    end
    Z{v} = constructAnchorDistance_PKN(T, m(v),k);
end
