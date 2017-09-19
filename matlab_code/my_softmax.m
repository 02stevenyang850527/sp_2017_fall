function res = my_softmax(X)
    % X is a 1-dim array, that is 1 x m
    eX = exp(X - max(X));
    res = eX./sum(eX);
    
end