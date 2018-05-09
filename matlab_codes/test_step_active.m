function [model, loss, acc, state_h, state_c, record]...
    = test_step_active(X_test,y_test,model,state_h,state_c)

    D = 39;  % input dimension
    H = 256;
    loss = 0;
    output_dim = 61;
    confident_th = 0.97;
    alpha = 0.001;
    
    % Forward part
    [sentence_size,~] = size(X_test); 
    y_pred = zeros(1, output_dim, sentence_size);
    
    cache_X = zeros(1, D, sentence_size);
    cache_h_old = zeros(1, H, sentence_size);
    cache_hf = zeros(1, H, sentence_size);
    cache_hi = zeros(1, H, sentence_size);
    cache_ho = zeros(1, H, sentence_size);
    cache_hc = zeros(1, H, sentence_size);
    
    Wf = model.Wf;
    Wi = model.Wi;
    Wo = model.Wo;
    Wc = model.Wc;
    Whf = model.Whf;
    Whi = model.Whi; 
    Who = model.Who;
    Whc = model.Whc;
    Wy = model.Wy;
    
    bf = model.bf;
    bi = model.bi;
    bo = model.bo;
    bc = model.bc;
    by = model.by;
    
    cache_c_old = zeros(1, H, sentence_size);
    cache_c = zeros(1, H, sentence_size);
    cache_h = zeros(1, H, sentence_size);
    
    record = zeros(1,sentence_size);
    
    for k=1:sentence_size
        [X,h_old,hf,hi,ho,hc,Wf,Whf,Wi,Whi,Wo,Who,Wc,Whc,c_old,state_c,Wy,y,state_h] ...
        = forward(Wf,Whf,Wi,Whi,Wc,Whc,Wo,Who,Wy,bf,bi,bc,bo,by,X_test(k,:),state_h,state_c,true);
        
        loss = loss + cross_entropy(y, y_test(k), 0, Wf, Wi, Wc, Wo, Wy);
        y_pred(:,:,k) = y;
        [~,record(k)] = max(y);
        cache_X(:,:,k) = X;
        cache_h_old(:,:,k) = h_old;
        cache_hf(:,:,k) = hf;
        cache_hi(:,:,k) = hi;
        cache_ho(:,:,k) = ho;
        cache_hc(:,:,k) = hc;      
        cache_c_old(:,:,k) = c_old;
        cache_h(:,:,k) = state_h;
        cache_c(:,:,k) = state_c;       
    end
    
    dh_next = zeros(1,H);
    dc_next = zeros(1,H);
    c_dWf = zeros(D,H);
    c_dWi = zeros(D,H);
    c_dWc = zeros(D,H);
    c_dWo = zeros(D,H);
    c_dWhf = zeros(1,H);
    c_dWhi = zeros(1,H);
    c_dWhc = zeros(1,H);
    c_dWho = zeros(1,H);
    c_dWy = zeros(H,output_dim);
    c_dbf = zeros(1,H);
    c_dbi = zeros(1,H);
    c_dbc = zeros(1,H);
    c_dbo = zeros(1,H);
    c_dby = zeros(1,output_dim);
    acc = sum(record==y_test);
    
    for k=1:sentence_size
        ind = sentence_size - k + 1;
        [dWf,dWhf,dWi,dWhi,dWc,dWhc,dWo,dWho,dWy,dbf,dbi,dbc,dbo,dby,dh_next,dc_next]= ...
        backward(y_pred(:,:,ind),y_test(ind),cache_X(:,:,ind),cache_h_old(:,:,ind),dh_next,dc_next,...
        cache_hf(:,:,ind),cache_hi(:,:,ind),cache_ho(:,:,ind),cache_hc(:,:,ind),...
        Wf, Whf, Wi,Whi, Wo, Who, Wc, Whc,cache_c_old(:,:,ind),cache_c(:,:,ind),Wy,cache_h(:,:,ind));
        cnt = 0;
        if ((y_test(ind) == record(ind)) && (max(y_pred(:,:,ind)) > confident_th))
            c_dWf = c_dWf + dWf;
            c_dWi = c_dWi + dWi;
            c_dWo = c_dWo + dWo;
            c_dWc = c_dWc + dWc;
            c_dWhf = c_dWhf + dWhf;
            c_dWhi = c_dWhi + dWhi;
            c_dWho = c_dWho + dWho;
            c_dWhc = c_dWhc + dWhc;
            c_dWy = c_dWy + dWy;
            c_dbf = c_dbf + dbf;
            c_dbi = c_dbi + dbi;
            c_dbo = c_dbo + dbo;
            c_dbc = c_dbc + dbc;
            c_dby = c_dby + dby;
            cnt = cnt + 1;
        end
        
        clip_value = 15;
        c_dWf = min(clip_value,c_dWf);
        c_dWf = max(-clip_value,c_dWf);
        c_dWi = min(clip_value,c_dWi);
        c_dWi = max(-clip_value,c_dWi);
        c_dWo = min(clip_value,c_dWo);
        c_dWo = max(-clip_value,c_dWo);
        c_dWc = min(clip_value,c_dWc);
        c_dWc = max(-clip_value,c_dWc);
        c_dWhf = min(clip_value,c_dWhf);
        c_dWhf = max(-clip_value,c_dWhf);
        c_dWhi = min(clip_value,c_dWhi);
        c_dWhi = max(-clip_value,c_dWhi);
        c_dWho = min(clip_value,c_dWho);
        c_dWho = max(-clip_value,c_dWho);
        c_dWhc = min(clip_value,c_dWhc);
        c_dWhc = max(-clip_value,c_dWhc);
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
    
        c_dWf = c_dWf/cnt;
        c_dWi = c_dWi/cnt;
        c_dWo = c_dWo/cnt;
        c_dWc = c_dWc/cnt;
        c_dWhf = c_dWhf/cnt;
        c_dWhi = c_dWhi/cnt;
        c_dWho = c_dWho/cnt;
        c_dWhc = c_dWhc/cnt;
        c_dWy = c_dWy/cnt;
        c_dbf = c_dbf/cnt;
        c_dbi = c_dbi/cnt;
        c_dbo = c_dbo/cnt;
        c_dbc = c_dbc/cnt;
        c_dby = c_dby/cnt;

        model.Wf = Wf - alpha .* c_dWf;
        model.Wi = Wi - alpha .* c_dWi;
        model.Wc = Wc - alpha .* c_dWc;
        model.Wo = Wo - alpha .* c_dWo;
        model.Whf = Whf - alpha .* c_dWhf;
        model.Whi = Whi - alpha .* c_dWhi;
        model.Whc = Whc - alpha .* c_dWhc;
        model.Who = Who - alpha .* c_dWho;
        model.bf = bf - alpha .* c_dbf;
        model.bi = bi - alpha .* c_dbi;
        model.bc = bc - alpha .* c_dbc;
        model.bo = bo - alpha .* c_dbo;
        model.Wy = Wy - alpha .* c_dWy;
        model.by = by - alpha .* c_dby;
    end
end
