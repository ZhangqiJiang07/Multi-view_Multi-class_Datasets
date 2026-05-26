function [tpr1,fpr1] = mlr_roc(f, y_test)

% [tpr,fpr] = mlr_roc(f, y_test)
% Calculates TPR (True Positive Rate) and FPR (False Positive Rate) and
% plots ROC curve.
%%% INPUTS %%%
% f       : matrix of ranking function outputs
% y_test  : label matrix for test set
%%% OUTPUTS %%%
% tpr : true positive rate values
% fpr : false positive rate values


K=size(y_test,2);

for i=1:K
     [match(:,i),fpp(:,i),fnn(:,i)] = performance(y_test,f,i);
    tp1(i)=sum(match(:,i));
    fn1(i)=sum(fnn(:,i));
    fp1(i)=sum(fpp(:,i));
    tn1(i)=K*size(f,1)-(tp1(i)+fp1(i)+fn1(i));
    tpr1(i)=tp1(i)/(tp1(i)+fn1(i)); % Samples are considered all-together
    fpr1(i)=fp1(i)/(fp1(i)+tn1(i));
end

% area1 = mlr_auc(fpr1,tpr1); % Samples are considered all-together

% for i=1:size(f,1)
%     tpr2=match(i,:)./(match(i,:)+fnn(i,:)); % Samples are considered one by one
%     fpr2=fpp(i,:)./(K-match(i,:)+fnn(i,:));
%     area2(i) = mlr_auc(fpr2,tpr2);
% end
% area2 = mean(area2);