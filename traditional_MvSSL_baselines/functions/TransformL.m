function InitF = TransformL(F, c)
F = F-1;
[n,~] = size(F);
InitF = zeros(n,c);
index = find(F==0);
InitF(index) = 1;
index = find(F==1);
InitF(index) = 1;

