function [X, hf, hi, ho, hc, Wf, Wi, Wo, Wc, c_old, c, Wy, y, h] ...
= forward(Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,X_one_hot,h_old,c_old,Train)
    
    X = [h_old,X_one_hot];
    
    hf = fc_forward(X, Wf, bf);
    hf = sigmoid_forward(hf);
    
    hi = fc_forward(X, Wi, bi);
    hi = sigmoid_forward(hi);
    
    ho = fc_forward(X, Wo, bo);
    ho = sigmoid_forward(ho);
    
    hc = fc_forward(X, Wc, bc);
    hc = tanh_forward(hc);
    
    c = hf .* c_old + hi .* hc;
    c = tanh_forward(c);
    
    h = ho .* c;
    
    y = fc_forward(h, Wy, by);
    
    if Train == false
        y = my_softmax(y);
    end
end