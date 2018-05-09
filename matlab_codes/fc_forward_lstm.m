function y = fc_forward_lstm(X,h,Wx,Wh,b)
    y = X*Wx + Wh.*h + b;
end
