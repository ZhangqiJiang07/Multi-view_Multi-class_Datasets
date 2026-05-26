function [M, Ll, Lu, Index, ic] = labelData(X,L,IndexO,L_ratio)
    % X: imcomplete multi-view feature matrices, cell, V nv*dv
    % L: the indicator of lables
    % Index: indicator of instances in each view, nxV
    % M: M=[Xl; Xu]
    % Ll: labeled instances indicator of gnd
    % Lu: unlabeled instances indicator of gnd
    
    View = max(size(X));
    C = max(L)-min(L)+1;
    nV = sum(IndexO,2);
    
    ic = [];
    for c = 1:C
        cc = find(L==c);
        ncl = ceil(max(size(cc))*L_ratio);
        nVc = nV(cc);
        for v = View:-1:1
            Ls = find(nVc==v);
            ts = size(Ls,1);
            if ts>ncl
                tc = randperm(ts,ncl);
                ic = [ic;cc(Ls(tc))];
                break;
            else
                ic = [ic;cc(Ls)];
                ncl = ncl-ts;
            end
        end
    end
    Ll = L(ic);
    L(ic)=[];
    Lu = L;
    
    Index = IndexO(ic,:);
    IndexO(ic,:) = [];
    Index = [Index;IndexO];
    
    % rearrange the position of labeled data
    for v = 1:View
        M{v} = X{v}(ic,:);
        X{v}(ic,:) = [];
        M{v} = [M{v};X{v}];
    end