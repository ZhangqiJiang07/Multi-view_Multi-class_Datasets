function [M] = bbcselectdata(X,ic,View)
if View>1
    for v = 1:View
        M{v} = X{v}(ic,:);
        X{v}(ic,:) = [];
        M{v} = [M{v};X{v}];
    end
else
    M = X(ic,:);
    X(ic,:) = [];
    M = [M;X];
end