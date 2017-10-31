function [dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, dh_next, dc_next] ...
    = backward(y_pred,y_train,X,dh_next,dc_next,hf,hi,ho,hc,Wf,Wi,Wo,Wc,c_old,c,Wy,h)

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
    
    [dXo, dWo, dbo] = fc_backward(dho, Wo, X);
    [dXc, dWc, dbc] = fc_backward(dhc, Wc, X);
    [dXi, dWi, dbi] = fc_backward(dhi, Wi, X);
    [dXf, dWf, dbf] = fc_backward(dhf, Wf, X);
    
     dX = dXo + dXc + dXi + dXf; %size(dX) = 1x135
     
     dh_next = dX(1:H);
     
     dc_next = hf .* dc;
     
end
