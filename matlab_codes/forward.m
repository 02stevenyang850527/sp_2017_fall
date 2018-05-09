function [X,h_old, hf, hi, ho, hc, Wf, Whf, Wi, Whi,Wo,Who, Wc,Whc, c_old, c, Wy, y, h] ...
= forward(Wf,Whf,Wi,Whi,Wc,Whc,Wo,Who,Wy,bf,bi,bc,bo,by,X,h_old,c_old,Train)
    
    
    hf = fc_forward_lstm(X, h_old, Wf, Whf, bf);
    hf = sigmoid_forward(hf);
    
    hi = fc_forward_lstm(X, h_old, Wi, Whi, bi);
    hi = sigmoid_forward(hi);
    
    ho = fc_forward_lstm(X, h_old, Wo, Who, bo);
    ho = sigmoid_forward(ho);
    
    hc = fc_forward_lstm(X, h_old, Wc, Whc, bc);
    hc = tanh_forward(hc);
    
    c = hf .* c_old + hi .* hc;
    %c = tanh_forward(c);
    
    h = ho .* tanh_forward(c);
    
    y = fc_forward(h, Wy, by);
    
    if Train == false
        y = my_softmax(y);
    end
end
