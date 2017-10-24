function dX = tanh_backward(dout, cache)
    dX = (1 - cache.^2) .* dout;
end