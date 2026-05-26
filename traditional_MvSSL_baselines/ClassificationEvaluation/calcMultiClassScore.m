function [Accuracy, F1, Precision] = calcMultiClassScore(pred_label, gt_label, classes)

cm = zeros(classes, classes);

if min(gt_label) == 0
    gt_label = gt_label + 1;
end

if min(pred_label) == 0
    pred_label = pred_label + 1;
end

for i = 1:numel(pred_label)
    cm(pred_label(i), gt_label(i)) = cm(pred_label(i), gt_label(i)) + 1;
end

Result_common = multiclass_metrics_common(cm);

Accuracy = Result_common.Accuracy;
F1 = Result_common.F1score;
Precision = Result_common.Precision;

end
