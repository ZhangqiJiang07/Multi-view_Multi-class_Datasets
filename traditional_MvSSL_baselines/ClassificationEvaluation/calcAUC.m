function [AUC] = calcAUC(f, label_test, classes)
%************************************************************************%
%
% Usage: metrics  = calcAUC(f, label, classes)
% Inputs: f  - matrix of ranking function outputs -- N x classes matrix
%         label_test - label matrix for test set
% 
% Outputs: AUC : the AUC performance

if sum(sum(f)) - size(f,1) < 1e-6
    rocZ = f;
else
    rocZ = f./sum(f,2);
end
% rocZ = f;
onehot_y = encodeOnehot(label_test, classes);
zero_index = find(sum(onehot_y, 2) == 0);
onehot_y(zero_index, :) = [];
rocZ(zero_index, :) = [];
newY = 2*onehot_y - ones(size(onehot_y));
[tpr, fpr] = mlr_roc(rocZ, newY);
[AUC, ~] = mlr_auc(fpr, tpr);



end