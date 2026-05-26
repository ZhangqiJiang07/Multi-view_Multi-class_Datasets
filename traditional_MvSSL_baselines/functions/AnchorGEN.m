% X:num*dim
%alpha:regulization num
%numanchor:the real number of anchors is 2^numAnchor
%[1] F. Nie, W. Zhu, and X. Li, "Unsupervised Large Graph Embedding," AAAI2017.
%[2] W. Zhu, F. Nie, and X. Li, "Fast Spectral Clustering with Efficient Large
%Graph Construction," ICASSP2017.
function [Z,M] = AGf(X,numAnchor,numNearestAnchor,selAnchor)

% BKHK 1
% K-means++ 2
% K-means 3
% Random 4
if nargin<4
    selAnchor = 1; % BKHK as default
end

[num,d] = size(X);

if selAnchor == 1 % BKHK
    [~,locAnchor] = hKM(X',[1 :num],numAnchor,1);
elseif selAnchor == 2 % kmeans++
    [~,locAnchor] = kmeans(X,2^(numAnchor),'Start','plus');
    locAnchor = locAnchor';
elseif selAnchor == 3 % kmeans
    [~,locAnchor] = kmeans(X,2^(numAnchor),'Start','sample');
    locAnchor = locAnchor';
elseif selAnchor == 4 % Random
    RandIDX = randperm(num,2^(numAnchor));
    locAnchor = X(RandIDX,:);locAnchor = locAnchor';
end
M=locAnchor';

Z = ConstructA_NP(X',(locAnchor),numNearestAnchor);
end