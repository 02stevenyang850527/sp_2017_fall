function [c_dWf,c_dWi,c_dWc,c_dWo,c_dWy,c_dbf,c_dbi,c_dbc,c_dbo,c_dby,...
          loss,state_h, state_c]...
    = train_step(X_train,y_train,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,state_h,state_c)

    loss = 0;
    % Forward part
    [batch_size, dim] = size(X_train); % X_train is in one-hot format
    
    y_pred = zeros(1, dim, batch_size);
    cache_X = zeros(1, 135, batch_size);
    cache_hf = zeros(1, 64, batch_size);
    cache_hi = zeros(1, 64, batch_size);
    cache_ho = zeros(1, 64, batch_size);
    cache_hc = zeros(1, 64, batch_size);
    cache_Wf = zeros(135, 64, batch_size);
    cache_Wi = zeros(135, 64, batch_size); 
    cache_Wo = zeros(135, 64, batch_size);
    cache_Wc = zeros(135, 64, batch_size);
    cache_c_old = zeros(1, 64, batch_size);
    cache_c = zeros(1, 64, batch_size);
    cache_Wy = zeros(64, 71, batch_size);
    cache_h = zeros(1, 64, batch_size);
    
    
    for k=1:batch_size
        [X,hf,hi,ho,hc,Wf,Wi,Wo,Wc,c_old,state_c,Wy,y,state_h] ...
        = forward(Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,X_train(k,:),state_h,state_c,true);
    
        loss = loss + cross_entropy(y, y_train(k), 0, Wf, Wi, Wc, Wo, Wy);
        y_pred(:,:,k) = y;
        cache_X(:,:,k) = X;
        cache_hf(:,:,k) = hf;
        cache_hi(:,:,k) = hi;
        cache_ho(:,:,k) = ho;
        cache_hc(:,:,k) = hc;
        cache_Wf(:,:,k) = Wf;
        cache_Wi(:,:,k) = Wi;
        cache_Wo(:,:,k) = Wo;
        cache_Wc(:,:,k) = Wc;       
        cache_c_old(:,:,k) = c_old;
        cache_Wy(:,:,k) = Wy;
        cache_h(:,:,k) = state_h;
        cache_c(:,:,k) = state_c;
        
    end
    loss = loss / batch_size;
    
    % Backward part
    dh_next = zeros(1,64);
    dc_next = zeros(1,64);
    c_dWf = zeros(135,64);
    c_dWi = zeros(135,64);
    c_dWc = zeros(135,64);
    c_dWo = zeros(135,64);
    c_dWy = zeros(64,71);
    c_dbf = zeros(1,64);
    c_dbi = zeros(1,64);
    c_dbc = zeros(1,64);
    c_dbo = zeros(1,64);
    c_dby = zeros(1,71);
    
    for k=1:batch_size
        ind = batch_size - k + 1;
        [dWf,dWi,dWc,dWo,dWy,dbf,dbi,dbc,dbo,dby,dh_next,dc_next]= ...
        backward(y_pred(:,:,ind),y_train(ind),cache_X(:,:,ind),dh_next,dc_next,...
        cache_hf(:,:,ind),cache_hi(:,:,ind),cache_ho(:,:,ind),cache_hc(:,:,ind),...
        cache_Wf(:,:,ind),cache_Wi(:,:,ind),cache_Wo(:,:,ind),cache_Wc(:,:,ind),...
        cache_c_old(:,:,ind),cache_c(:,:,ind),cache_Wy(:,:,ind),cache_h(:,:,ind));
        
        c_dWf = c_dWf + dWf;
        c_dWi = c_dWi + dWi;
        c_dWo = c_dWo + dWo;
        c_dWc = c_dWc + dWc;
        c_dWy = c_dWy + dWy;
        c_dbf = c_dbf + dbf;
        c_dbi = c_dbi + dbi;
        c_dbo = c_dbo + dbo;
        c_dbc = c_dbc + dbc;
        c_dby = c_dby + dby;
        
    end
    
    % clip the gradient value
    c_dWf = min(5,c_dWf);
    c_dWf = max(-5,c_dWf);
    c_dWi = min(5,c_dWi);
    c_dWi = max(-5,c_dWi);
    c_dWo = min(5,c_dWo);
    c_dWo = max(-5,c_dWo);
    c_dWc = min(5,c_dWc);
    c_dWc = max(-5,c_dWc);
    c_dWy = min(5,c_dWy);
    c_dWy = max(-5,c_dWy);
    c_dbf = min(5,c_dbf);
    c_dbf = max(-5,c_dbf);
    c_dbi = min(5,c_dbi);
    c_dbi = max(-5,c_dbi);
    c_dbo = min(5,c_dbo);
    c_dbo = max(-5,c_dbo);
    c_dbc = min(5,c_dbc);
    c_dbc = max(-5,c_dbc);
    c_dby = min(5,c_dby);
    c_dby = max(-5,c_dby);
    
end
    