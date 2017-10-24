function [dX,dW,db] = fc_backward(dout,cache_W,cache_h)
    dW = (cache_h.') * dout;
    db = dout;
    dX = dout * (cache_W.');
    
end