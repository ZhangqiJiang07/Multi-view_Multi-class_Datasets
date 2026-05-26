function [Result,ReferenceResult] = multiclass_metrics_special(confMatrix)
            
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
%     %confusion matrix for multiple class start
%     %Inputs-1.Actual Class Labels,2.Predict Class Labels and 3.Display if need
%     %Outputs
%     
%     %1.Result-Struct Over all output Which has follwing
%     %2.RefereceResult indidual output Which has follwing
%     %%%%%%%%1.acuuracy
%     %%%%%%%%2.error
%     %%%%%%%%3.Recall (Recall or True positive rate)
%     %%%%%%%%4.Specificity
%     %%%%%%%%5.Precision
%     %%%%%%%%6.FPR-False positive rate
%     %%%%%%%%7.F_score
%     %%%%%%%%8.MCC-Matthews correlation coefficient
%     %%%%%%%%9.kappa-Cohen's kappa
% 
%     %%Original Developer Mr.Abbas Manthiri S
%     %%Date  25-12-2016
%     %%Mail Id: abbasmanthiribe@gmail.com
%     %%http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
%     %%https://en.wikipedia.org/wiki/Confusion_matrix
%     
%     %%Modified by Preetham Manjunatha
%     %%Date  02-03-2020
%     %%Note: Fixed Abbas written code where the NaN and Inf values were
%     not considered while performaing the class average.
% 
%     % clc
%     % clear all
%     % close all
%     % %%Multiclass or two class
%     % [Result,RefereceResult] = multiclass_metrics(confMatrix)
            
    % Size of confusion matrix
    [row,col]=size(confMatrix);
    if row~=col
        error('Confusion matrix dimention is wrong')
    end
    
    % Number of classes
    n_class=row;
    
    % Find TP, FN, FP and TN
    switch n_class
        case 2
            TP=confMatrix(1,1);
            FN=confMatrix(1,2);
            FP=confMatrix(2,1);
            TN=confMatrix(2,2);

        otherwise
            TP=zeros(1,n_class);
            FN=zeros(1,n_class);
            FP=zeros(1,n_class);
            TN=zeros(1,n_class);
            for i=1:n_class
                TP(i)=confMatrix(i,i);
                FN(i)=sum(confMatrix(i,:))-confMatrix(i,i);
                FP(i)=sum(confMatrix(:,i))-confMatrix(i,i);
                TN(i)=sum(confMatrix(:))-TP(i)-FP(i)-FN(i);
            end

    end
            
    %% Calulations
    %1.P-Positive
    %2.N-Negative
    %3.acuuracy
    %4.error
    %5.Recall (Recall or True positive rate)
    %6.Specificity
    %7.Precision
    %8.FPR-False positive rate
    %9.F_score
    %10.MCC-Matthews correlation coefficient
    %11.kappa-Cohen's kappa
    P=TP+FN;
    N=FP+TN;
    switch n_class
        case 2
            accuracy=(TP+TN)/(P+N);
            Error=1-accuracy;
            Result.Accuracy=(accuracy);
            Result.Error=(Error);
        otherwise
            accuracy=(TP)./(P+N);
            Error=(FP)./(P+N);
            Result.Accuracy=sum(accuracy);
            Result.Error=sum(Error);
    end
    
    ReferenceResult.AccuracyOfSingle=(TP ./ P)';
    ReferenceResult.ErrorOfSingle=1-ReferenceResult.AccuracyOfSingle;
    Recall=TP./P;
    Specificity=TN./N;
    Precision=TP./(TP+FP);
    FPR=1-Specificity;
    beta=1;
    F1_score=( (1+(beta^2))*(Recall.*Precision) ) ./ ( (beta^2)*(Precision+Recall) );
    MCC=[( TP.*TN - FP.*FN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) );...
        ( FP.*FN - TP.*TN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) )] ;
    MCC=max(MCC);
    
    %Kappa Calculation BY 2x2 Matrix Shape
    pox=sum(accuracy);
    Px=sum(P);TPx=sum(TP);FPx=sum(FP);TNx=sum(TN);FNx=sum(FN);Nx=sum(N);
    pex=( (Px.*(TPx+FPx))+(Nx.*(FNx+TNx)) ) ./ ( (TPx+TNx+FPx+FNx).^2 );
    kappa_overall=([( pox-pex ) ./ ( 1-pex );( pex-pox ) ./ ( 1-pox )]);
    kappa_overall=max(kappa_overall);

    %Kappa Calculation BY n_class x n_class Matrix Shape
    po=accuracy;
    pe=( (P.*(TP+FP))+(N.*(FN+TN)) ) ./ ( (TP+TN+FP+FN).^2 );
    kappa=([( po-pe ) ./ ( 1-pe );( pe-po ) ./ ( 1-po )]);
    kappa=max(kappa);


    %%
    %Output Struct for individual Classes
    %  RefereceResult.Class=class_ref;
    ReferenceResult.AccuracyInTotal=accuracy';
    ReferenceResult.ErrorInTotal=Error';
    ReferenceResult.Recall=Recall';
    ReferenceResult.Specificity=Specificity';
    ReferenceResult.Precision=Precision';
    ReferenceResult.FalsePositiveRate=FPR';
    ReferenceResult.F1_score=F1_score';
    ReferenceResult.MatthewsCorrelationCoefficient=MCC';
    ReferenceResult.Kappa=kappa';
    ReferenceResult.TruePositive=TP';
    ReferenceResult.FalsePositive=FP';
    ReferenceResult.FalseNegative=FN';
    ReferenceResult.TrueNegative=TN';

    % Remove NANs and INFs
    stats = [Precision', Recall', F1_score', MCC'];
    stats(any(isinf(stats),2),:) = 0;
    stats(any(isnan(stats),2),:) = 0;

    %Output Struct for over all class lists
    Result.Recall=mean(stats(:,2));
    Result.Specificity=mean(Specificity);
    Result.Precision=mean(stats(:,1));
    Result.FalsePositiveRate=mean(FPR);
    Result.F1_score=mean(stats(:,3));
    Result.MatthewsCorrelationCoefficient=mean(stats(:,4));
    Result.Kappa=kappa_overall;

end