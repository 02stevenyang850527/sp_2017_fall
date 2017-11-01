function solver(X_train,Y_train,alpha,batch_size,n_iter,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by)
    beta1 = 0.9;
    beta2 = 0.999;
    print_after = 1000;
    
    cnt = 0;  % count for batch_size 
    state_h = zeros(1,64);
    state_c = zeros(1,64);
    [m,n] = size(X_train);
    smooth_loss = -log(1 / n);
    num_batch = ceil(m/batch_size);
    M_Wf = zeros(135,64);
    M_Wi = zeros(135,64);
    M_Wc = zeros(135,64);
    M_Wo = zeros(135,64);
    M_Wy = zeros(64,71);
    M_bf = zeros(1,64);
    M_bi = zeros(1,64);
    M_bc = zeros(1,64);
    M_bo = zeros(1,64);
    M_by = zeros(1,71);
        
    R_Wf = zeros(135,64);
    R_Wi = zeros(135,64);
    R_Wc = zeros(135,64);
    R_Wo = zeros(135,64);
    R_Wy = zeros(64,71);
    R_bf = zeros(1,64);
    R_bi = zeros(1,64);
    R_bc = zeros(1,64);
    R_bo = zeros(1,64);
    R_by = zeros(1,71);
    
    for iter = 1:n_iter
        t = iter;
        if cnt >= num_batch
            cnt = 0;
            state_h = zeros(1,64);
            state_c = zeros(1,64);
        end
        
        if cnt == num_batch-1
            X_mini = X_train(cnt*batch_size+1:end,:);
            Y_mini = Y_train(cnt*batch_size+1:end);
        else
            X_mini = X_train(cnt*batch_size+1:(cnt+1)*batch_size,:);
            Y_mini = Y_train(cnt*batch_size+1:(cnt+1)*batch_size);
        end
        
        cnt = cnt + 1;
        
        if mod(iter, print_after) == 0
            disp("=======================================");
            fprintf("Iter: %d, loss: %f\n" ,int64(iter),smooth_loss);
            disp("=======================================");
            %sample(X_mini(1,:),100,word,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,state_h,state_c);
            disp(' ');
        end
        
        [c_dWf,c_dWi,c_dWc,c_dWo,c_dWy,c_dbf,c_dbi,c_dbc,c_dbo,c_dby,...
         loss,state_h, state_c]...
        = train_step(X_mini,Y_mini,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,state_h,state_c);
        
        
        
        smooth_loss = 0.999 .*smooth_loss + 0.001 .* loss;
        
        M_Wf = exp_running_avg(M_Wf, c_dWf, beta1);
        M_Wi = exp_running_avg(M_Wi, c_dWi, beta1);
        M_Wc = exp_running_avg(M_Wc, c_dWc, beta1);
        M_Wo = exp_running_avg(M_Wo, c_dWo, beta1);
        M_Wy = exp_running_avg(M_Wy, c_dWy, beta1);
        M_bf = exp_running_avg(M_bf, c_dbf, beta1);
        M_bi = exp_running_avg(M_bi, c_dbi, beta1);
        M_bc = exp_running_avg(M_bc, c_dbc, beta1);
        M_bo = exp_running_avg(M_bo, c_dbo, beta1);
        M_by = exp_running_avg(M_by, c_dby, beta1);
        
        R_Wf = exp_running_avg(R_Wf, c_dWf.^2, beta2);
        R_Wi = exp_running_avg(R_Wi, c_dWi.^2, beta2);
        R_Wc = exp_running_avg(R_Wc, c_dWc.^2, beta2);
        R_Wo = exp_running_avg(R_Wo, c_dWo.^2, beta2);
        R_Wy = exp_running_avg(R_Wy, c_dWy.^2, beta2);
        R_bf = exp_running_avg(R_bf, c_dbf.^2, beta2);
        R_bi = exp_running_avg(R_bi, c_dbi.^2, beta2);
        R_bc = exp_running_avg(R_bc, c_dbc.^2, beta2);
        R_bo = exp_running_avg(R_bo, c_dbo.^2, beta2);
        R_by = exp_running_avg(R_by, c_dby.^2, beta2);
        
        m_Wf_hat = M_Wf ./ (1 - beta1^t);
        m_Wi_hat = M_Wi ./ (1 - beta1^t);
        m_Wc_hat = M_Wc ./ (1 - beta1^t);
        m_Wo_hat = M_Wo ./ (1 - beta1^t);
        m_Wy_hat = M_Wy ./ (1 - beta1^t);
        m_bf_hat = M_bf ./ (1 - beta1^t);
        m_bi_hat = M_bi ./ (1 - beta1^t);
        m_bc_hat = M_bc ./ (1 - beta1^t);
        m_bo_hat = M_bo ./ (1 - beta1^t);
        m_by_hat = M_by ./ (1 - beta1^t);
        
        r_Wf_hat = R_Wf ./ (1 - beta2^t);
        r_Wi_hat = R_Wi ./ (1 - beta2^t);
        r_Wc_hat = R_Wc ./ (1 - beta2^t);
        r_Wo_hat = R_Wo ./ (1 - beta2^t);
        r_Wy_hat = R_Wy ./ (1 - beta2^t);
        r_bf_hat = R_bf ./ (1 - beta2^t);
        r_bi_hat = R_bi ./ (1 - beta2^t);
        r_bc_hat = R_bc ./ (1 - beta2^t);
        r_bo_hat = R_bo ./ (1 - beta2^t);
        r_by_hat = R_by ./ (1 - beta2^t);
        
        Wf = Wf - alpha .* m_Wf_hat ./ (sqrt(r_Wf_hat) + 1e-8);
        Wi = Wi - alpha .* m_Wi_hat ./ (sqrt(r_Wi_hat) + 1e-8);
        Wc = Wc - alpha .* m_Wc_hat ./ (sqrt(r_Wc_hat) + 1e-8);
        Wo = Wo - alpha .* m_Wo_hat ./ (sqrt(r_Wo_hat) + 1e-8);
        Wy = Wy - alpha .* m_Wy_hat ./ (sqrt(r_Wy_hat) + 1e-8);
        bf = bf - alpha .* m_bf_hat ./ (sqrt(r_bf_hat) + 1e-8);
        bi = bi - alpha .* m_bi_hat ./ (sqrt(r_bi_hat) + 1e-8);
        bc = bc - alpha .* m_bc_hat ./ (sqrt(r_bc_hat) + 1e-8);
        bo = bo - alpha .* m_bo_hat ./ (sqrt(r_bo_hat) + 1e-8);
        by = by - alpha .* m_by_hat ./ (sqrt(r_by_hat) + 1e-8);
       

    end
    
end