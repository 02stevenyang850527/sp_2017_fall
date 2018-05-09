function result = sigmoid_backward(dout, cache)
    
    result = cache .* (1-cache) .* dout;
    
end
