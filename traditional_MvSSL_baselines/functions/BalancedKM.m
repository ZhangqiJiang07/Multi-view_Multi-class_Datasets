function [C, Q, y] = BalancedKM(X, ratio, InitF)

class_num = 2;
n = size(X,2);

if nargin < 2
    ratio = 0.5;
end
if nargin < 3
%     StartInd = repmat([1:class_num]',ceil(n/class_num),1);
%     StartInd=StartInd(1:n);
    StartInd = randsrc(n,1,1:class_num);
    InitF = TransformL(StartInd, class_num); % convert the class num to one-hot encode
end

if ratio > 0.5
    error('ratio should not larger than 0.5');
end
if ratio < 0
    ratio = 0;
end

a = floor(n*ratio);
b = floor(n*(1-ratio));

F = InitF;
for iter = 1:10
    C = X*F./sum(F+eps);            % calculate the centers for two categories
    F = sparse(n,class_num);
    Q = EuDist2(X',C',0);
    q = Q(:,1)-Q(:,2);
    [temp, idx] = sort(q);
    nn = length(find(temp<0));      % find the nearest cluster center for points 
    if nn>=a && nn<=b
        cp = nn;
    elseif nn<a
        cp = a;
    else
        cp = b;
    end;
    
    if cp < 1
        cp = 1;
    elseif cp > n-1
        cp = n-1;
    end;
%     n
%     size(F)
%     size(idx)
%     cp
    F(idx(1:cp),1) = 1;
    F(:,2) = 1-F(:,1);
    
%     XC = X-C*F';
%     obj(iter) = trace(XC*XC');
end;

[~, y] = max(F,[],2);