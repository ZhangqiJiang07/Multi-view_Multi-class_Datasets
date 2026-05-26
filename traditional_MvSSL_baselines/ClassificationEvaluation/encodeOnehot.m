function [onehot_y] = encodeOnehot(label, classes)

    onehot_y = zeros(max(size(label)), classes);
    for i = 1:max(size(label))
        onehot_y(i, label(i)) = 1;
    end
    
end