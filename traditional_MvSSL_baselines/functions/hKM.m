function [ys, centers] = hKM(X, idx0, k, count)

count=1;

n = size(X,2);

X0 = X(:,idx0);
if k == 1
    [centers, ~, y] = BalancedKM(X0,0.0);
else
    [centers, ~, y] = BalancedKM(X0,0.5);
end
ys = 2*count+1-y;
if k > 1
    id1 = find(y==1);
    idx1 = idx0(id1);
    [ys1, centers1] = hKM(X,idx1,k-1,2*count-1);

    id2 = find(y==2);
    idx2 = idx0(id2);
    [ys2, centers2] = hKM(X,idx2,k-1,2*count);

    ys(id1) = ys1;
    ys(id2) = ys2;
    centers = [centers1,centers2];
end