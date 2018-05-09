function [dWf,dWhf, dWi,dWhi, dWc,dWhc, dWo,dWho, dWy, dbf, dbi, dbc, dbo, dby, dh_next, dc_next] ...
    = backward(y_pred,y_train,X,h_old,dh_next,dc_next,hf,hi,ho,hc,Wf,Whf,Wi,Whi,Wo,Who,Wc,Whc,c_old,c,Wy,h)

    H = 256;
    dy = dcross_entropy(y_pred, y_train);
    [dh,dWy,dby] = fc_backward(dy, Wy, h);
    
    dh = dh + dh_next;
    
    dho = tanh(c) .* dh;
    dho = sigmoid_backward(dho, ho);
    
    dc = ho .* dh;
    dc = tanh_backward(dc, tanh(c));
    dc = dc + dc_next;
    
    dhf = c_old .* dc;
    dhf = sigmoid_backward(dhf, hf);
    
    dhi = hc .* dc;
    dhi = sigmoid_backward(dhi, hi);
    
    dhc = hi .* dc;
    dhc = tanh_backward(dhc, hc);
    
    [dXo, dWo, dWho, dbo, dho] = fc_backward_lstm(dho, Wo, Who, X, h_old);
    [dXc, dWc, dWhc, dbc, dhc] = fc_backward_lstm(dhc, Wc, Whc, X, h_old);
    [dXi, dWi, dWhi, dbi, dhi] = fc_backward_lstm(dhi, Wi, Whi, X, h_old);
    [dXf, dWf, dWhf, dbf, dhf] = fc_backward_lstm(dhf, Wf, Whf, X, h_old);
    
     dX = dXo + dXc + dXi + dXf;
     
     dh_next = dho + dhc + dhi + dhf;
     
     dc_next = hf .* dc;
     
end
