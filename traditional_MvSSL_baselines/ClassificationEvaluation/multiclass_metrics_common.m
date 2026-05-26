function metrics = multiclass_metrics_common(confmat) 

%%***********************************************************************%
%*                         Multiclass metrics                           *%
%*        Finds the multiclss metrics provided a confusion matrix.      *%
%*                                                                      *%
%* Code author: Preetham Manjunatha                                     *%
%* Github link: https://github.com/preethamam                           %*
%* Date: 11/24/2021                                                     *%
%************************************************************************%
%
%************************************************************************%
%
% Usage: metrics  = multiclass_metrics_common(confmat)
% Inputs: confmat  - confusion matrix -- N x N matrix
% 
% Outputs: metrics - metrics.Precision = Precision;
%                    metrics.Recall = Recall;
%                    metrics.Accuracy = Accuracy;
%                    metrics.Specificity = Specificity;
%                    metrics.F1score = F1score;

% Array initialization
N = size(confmat,1);

Precision = zeros(1,N);
Recall = zeros(1,N);
Specificity = zeros(1,N);
Accuracy = zeros(1,N);
F1score = zeros(1,N);


if size(confmat,1) > 2

     for i = 1:size(confmat,1)

        TP = confmat(i,i);
        FN = sum(confmat(i,:))-confmat(i,i);
        FP = sum(confmat(:,i))-confmat(i,i);
        TN = sum(confmat(:))-TP -FP-FN;

        Precision(:,i)   = TP / (TP+FP); % positive predictive value (PPV)
        Recall(:,i)      = TP / (TP+FN); % true positive rate (TPR), sensitivity
        if ((TN / (TN+FP)) > 1)
            Specificity(:,i) = 1;
        elseif ((TN / (TN+FP)) < 0)
            Specificity(:,i) = 0;
        else
            Specificity(:,i) = TN / (TN+FP); % (SPC) or true negative rate
        end
        Accuracy(:,i)    = (TP)/(TP+TN+FP+FN); % Accuracy
        F1score(:,i)     = (2*TP) /(2*TP + FP + FN);
     end
    
        % Remove junks         
        stats = [Precision', Recall', F1score', Accuracy', Specificity'];
        stats(any(isinf(stats),2),:) = 0;
        stats(any(isnan(stats),2),:) = 0;
        
        % Compute averages
        Accuracy  = sum(stats(:,4));
        Precision = mean(stats(:,1));
        Recall    = mean(stats(:,2));
        Specificity = mean(Specificity);
        F1score = mean(stats(:,3));
else
        TP = confmat(1, 1);
        FP = confmat(2, 1);
        FN = confmat(1, 2);
        TN = confmat(2,2);

        Precision = TP / (TP+FP); % positive predictive value (PPV)
        Recall    = TP / (TP+FN); % true positive rate (TPR), sensitivity
        if ((TN / (TN+FP)) > 1)
            Specificity = 1;
        elseif ((TN / (TN+FP)) < 0)
            Specificity = 0;
        else
            Specificity = TN / (TN+FP); % (SPC) or true negative rate
        end
        Accuracy = (TP+TN)/(TP+TN+FP+FN); % Accuracy
        F1score  = 2*TP /(2*TP + FP + FN);
    
end

% Output
metrics.Precision = Precision;
metrics.Recall = Recall;
metrics.Accuracy = Accuracy;
metrics.Specificity = Specificity;
metrics.F1score = F1score;
 

