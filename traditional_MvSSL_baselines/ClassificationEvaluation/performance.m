function [match,fp,fn] = performance(y,f,T)
% [match,fp,fn] = performance(y,f,T)
% finds the number of matches, fp and fn for the top T ranking values.
%%% INPUTS %%%
% y = labels
% f = ranking function outputs
% T = number of predicted labels
%%% OUTPUTS %%%
% match : is a vector which contains the number of hits for each instance in the test data set
% fp    : vector containing number of FPs (false positives) for each instance
% fn    : vector containing number of FNs (false negatives) for each instance


n=size(f,1);
K=size(f,2);

match=zeros(n,1);
fn=zeros(n,1);
fp=zeros(n,1);
clear pred_labels;

if n > 1000
        parfor i=1:n
            [~, si]=sort(f(i,:),'descend'); % for the i'th sample, rank its labels
            words=y(i,:);
            correct_labels=find(words>-1); % the ground-truth positive labels of test data
            si=si(1:T);  % the first T labels are considered as positive labels
            match(i)=0;
            for j=1: length(correct_labels)                     %max(size(correct_labels))
                if(find(si==correct_labels(j)))
                    match(i)=match(i)+1;
                end
            end
            fn(i)=length(correct_labels)-match(i);
            fp(i)=T-match(i);
        end
else
       for i=1:n
            [~, si]=sort(f(i,:),'descend'); % for the i'th sample, rank its labels
            words=y(i,:);
            correct_labels=find(words>-1); % the ground-truth positive labels of test data
            si=si(1:T);  % the first T labels are considered as positive labels
            match(i)=0;
            for j=1: length(correct_labels)                     %max(size(correct_labels))
                if(find(si==correct_labels(j)))
                    match(i)=match(i)+1;
                end
            end
            fn(i)=length(correct_labels)-match(i);
            fp(i)=T-match(i);
       end
end
    