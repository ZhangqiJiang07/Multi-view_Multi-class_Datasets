function [X,O,dim]=cdata(M,Index)
View = max(size(M));
X = [];
O = [];
for v = 1:View
    M{v}(isnan(M{v})==1) = 0;
    X = [X,M{v}];
    dim(v) = size(M{v},2);
    O = [O,repmat(Index(:,v),1,dim(v))];
end