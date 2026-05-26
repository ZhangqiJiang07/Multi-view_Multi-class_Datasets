function [X] = sdata(M,dim)
View = max(size(dim));
for v = 1:View
    X{v} = M(:,1:dim(v));
    M(:,1:dim(v)) = [];
end
