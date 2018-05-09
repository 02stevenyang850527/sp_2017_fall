function grad_y = dcross_entropy(y_pred, y_train)
    % Notes:
    % 1. y_pred is in one hot format, in this case: size = 1x61
    % 2. y_train is an index, in this case: from 1~61
    grad_y = my_softmax(y_pred);
    grad_y(y_train) = grad_y(y_train) - 1;
    
end
