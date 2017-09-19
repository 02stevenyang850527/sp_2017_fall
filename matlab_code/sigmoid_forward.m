function out = sigmoid_forward(X)
    out = 1 ./ (1+exp(-X));
end