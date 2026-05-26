function [M,Index] = partialData(X,P_ratio,mode)
% X: multi-view feature matrices, cell, V n*dv
% P_ratio: missing ratio of 
% M: imcomplete multi-view feature matrices, cell, V nv*dv
% Index: indicator of instances in each view, nxV
if nargin < 3
    mode = 1;
end


V = max(size(X));
num = size(X{1},1);
Index = ones(num,V);
P_r = 100*P_ratio;
M = X;
for i = 1:num
    for v = 1:V
        aa = randperm(100,1);
        if aa<=P_r
            M{v}(i,:) = 0; % nan
            Index(i,v) = 0;
        end
    end
    if sum(Index(i,:))==0
        bb = randperm(v,1);
        M{bb}(i,:) = X{bb}(i,:);
        Index(i,bb) = 1;
    end
end

%% Normalization 
for v = 1:V
    T1 = find(Index(:,v) == 1);
    T2 = M{v}(T1,:);
    if mode == 1 % Min-Max
        aa = max(max(T2));
        bb = min(min(T2));
        M{v}(T1,:) = (T2-bb) / max(1e-12, aa-bb);
    elseif mode == 2 % normalization
        for i = 1:length(T1)
            M{v}(T1(i),:) = T2(i,:) ./ max(1e-12, norm(T2(i,:)));
        end
    end
end

end