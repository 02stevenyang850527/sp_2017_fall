function [c_dWf,c_dWi,c_dWc,c_dWo,c_dWy,c_dbf,c_dbi,c_dbc,c_dbo,c_dby,...
          loss,acc,state_h, state_c, record]...
    = train_step(X_train,y_train,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,state_h,state_c)

    % Function: train one input character of sentence each time
    
    D = 39;  % input dimension
    H = 256;
    loss = 0;
    output_dim = 48;
    % Forward part
    [sentence_size,~] = size(X_train); % X_train is in one-hot format
    y_pred = zeros(1, output_dim, sentence_size);
    cache_X = zeros(1, D+H, sentence_size);
    cache_hf = zeros(1, H, sentence_size);
    cache_hi = zeros(1, H, sentence_size);
    cache_ho = zeros(1, H, sentence_size);
    cache_hc = zeros(1, H, sentence_size);
    cache_Wf = zeros(D+H, H, sentence_size);
    cache_Wi = zeros(D+H, H, sentence_size); 
    cache_Wo = zeros(D+H, H, sentence_size);
    cache_Wc = zeros(D+H, H, sentence_size);
    cache_c_old = zeros(1, H, sentence_size);
    cache_c = zeros(1, H, sentence_size);
    cache_Wy = zeros(H, output_dim, sentence_size);
    cache_h = zeros(1, H, sentence_size);
    
    record = zeros(1,sentence_size);
    for k=1:sentence_size
        [X,hf,hi,ho,hc,Wf,Wi,Wo,Wc,c_old,state_c,Wy,y,state_h] ...
        = forward(Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,X_train(k,:),state_h,state_c,true);
        
        loss = loss + cross_entropy(y, y_train(k), 0, Wf, Wi, Wc, Wo, Wy);
        y_pred(:,:,k) = y;
        [~,record(k)] = max(y);
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
    
    
    % Backward part
    dh_next = zeros(1,H);
    dc_next = zeros(1,H);
    c_dWf = zeros(D+H,H);
    c_dWi = zeros(D+H,H);
    c_dWc = zeros(D+H,H);
    c_dWo = zeros(D+H,H);
    c_dWy = zeros(H,output_dim);
    c_dbf = zeros(1,H);
    c_dbi = zeros(1,H);
    c_dbc = zeros(1,H);
    c_dbo = zeros(1,H);
    c_dby = zeros(1,output_dim);
   
    acc = sum(record==y_train);
    for k=1:sentence_size
        ind = sentence_size - k + 1;
        
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
    clip_value = 15;
    c_dWf = min(clip_value,c_dWf);
    c_dWf = max(-clip_value,c_dWf);
    c_dWi = min(clip_value,c_dWi);
    c_dWi = max(-clip_value,c_dWi);
    c_dWo = min(clip_value,c_dWo);
    c_dWo = max(-clip_value,c_dWo);
    c_dWc = min(clip_value,c_dWc);
    c_dWc = max(-clip_value,c_dWc);
    c_dWy = min(clip_value,c_dWy);
    c_dWy = max(-clip_value,c_dWy);
    c_dbf = min(clip_value,c_dbf);
    c_dbf = max(-clip_value,c_dbf);
    c_dbi = min(clip_value,c_dbi);
    c_dbi = max(-clip_value,c_dbi);
    c_dbo = min(clip_value,c_dbo);
    c_dbo = max(-clip_value,c_dbo);
    c_dbc = min(clip_value,c_dbc);
    c_dbc = max(-clip_value,c_dbc);
    c_dby = min(clip_value,c_dby);
    c_dby = max(-clip_value,c_dby);
         
end
