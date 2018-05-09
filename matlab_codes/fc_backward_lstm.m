function [dX,dWx,dWh,db,dh] = fc_backward_lstm(dout, Wx, Wh, x,h)
    db = dout;
    dWx = (x.') * dout;
    dWh = h .* dout;
    dh = Wh .* dout;
    dX = dout * (Wx.');
end
