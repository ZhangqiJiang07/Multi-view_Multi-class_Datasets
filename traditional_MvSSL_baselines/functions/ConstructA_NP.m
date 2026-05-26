

function A = ConstructA_NP(TrainData, Anchor,k)
%d*n
Dis = EuDist2(TrainData',Anchor',0);
% Dis = Dis + eps*ones(size(Dis,1),size(Dis,2));
[~,idx] = sort(Dis,2); %3.673212 seconds.
idx1 = idx(:,1:k+1);
clear idx;
[~,anchor_num] = size(Anchor);
[~,num] = size(TrainData);
A = zeros(num,anchor_num);
for i = 1:num
    id = idx1(i,1:k+1);
    di = Dis(i,id);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;
A = sparse(A);

