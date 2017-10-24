function [output_loss,output_acc] = solver(X_train,Y_train,alpha,batch_size,n_iter,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by)
    beta1 = 0.9;
    beta2 = 0.999;
    print_after = 1;
    H = 256;
    D = 39;  % number of MFCC features
    output_dim = 48;
    
    cnt = 0;  % count for batch_size 
    m = size(X_train,2);
    smooth_loss = -log(1 / D);
    output_loss = zeros(1,floor(n_iter/print_after));
    output_acc = zeros(1,floor(n_iter/print_after));
    num_batch = ceil(m/batch_size);
    
    M_Wf = zeros(D+H,H);
    M_Wi = zeros(D+H,H);
    M_Wc = zeros(D+H,H);
    M_Wo = zeros(D+H,H);
    M_Wy = zeros(H,output_dim);
    M_bf = zeros(1,H);
    M_bi = zeros(1,H);
    M_bc = zeros(1,H);
    M_bo = zeros(1,H);
    M_by = zeros(1,output_dim);
        
    R_Wf = zeros(D+H,H);
    R_Wi = zeros(D+H,H);
    R_Wc = zeros(D+H,H);
    R_Wo = zeros(D+H,H);
    R_Wy = zeros(H,output_dim);
    R_bf = zeros(1,H);
    R_bi = zeros(1,H);
    R_bc = zeros(1,H);
    R_bo = zeros(1,H);
    R_by = zeros(1,output_dim);
    
    for iter = 1:n_iter
        t = iter;
        
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
        c_loss = 0;
        total = 0;
        acc = 0;
        
        if cnt >= num_batch
            cnt = 0;
        end
        
        if cnt == num_batch-1
            for batch_cnt = 1: mod(m,batch_size)
                X = X_train{cnt*batch_size + batch_cnt};
                Y = Y_train{cnt*batch_size + batch_cnt};
                state_h = zeros(1,H);
                state_c = zeros(1,H);
                total = total + size(X,1);
   
                [dWf,dWi,dWc,dWo,dWy,dbf,dbi,dbc,dbo,dby,loss,acc,~,~,record]...
                = train_step(X,Y,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,state_h,state_c,acc);
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
                c_loss = c_loss + loss;
            end
        else
            for batch_cnt = 1:batch_size
                X = X_train{cnt*batch_size + batch_cnt};
                Y = Y_train{cnt*batch_size + batch_cnt};
                state_h = zeros(1,H);
                state_c = zeros(1,H);
                total = total + size(X,1);
   
                [dWf,dWi,dWc,dWo,dWy,dbf,dbi,dbc,dbo,dby,loss,acc,~,~,record]...
                = train_step(X,Y,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,state_h,state_c,acc);
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
                c_loss = c_loss + loss;
            end
        end
        
        cnt = cnt + 1; % finish one batch
        
        
        if (mod(iter,5)==0)
            disp('==');
            disp(record(56:66));
            disp(Y(56:66));
            disp('==');
        end
        
        c_loss = c_loss*200/total;  % average loss of 200 frame-based sequence input
        acc = acc / total;
        smooth_loss = 0.999 .* smooth_loss + 0.001 .* c_loss;
        
        if mod(iter, print_after) == 0
            output_loss(iter/print_after) = c_loss;
            output_acc(iter/print_after) = acc;
            disp('======================================');
            disp(strcat('Iter: ',num2str(iter),', loss: ',num2str(c_loss)));
            disp(strcat('acc: ',num2str(acc)));
            disp('======================================');
            disp(' ');
        end

        c_dWf = c_dWf/total;
        c_dWi = c_dWi/total;
        c_dWc = c_dWc/total;
        c_dWo = c_dWo/total;
        c_dWy = c_dWy/total;
        c_dbf = c_dbf/total;
        c_dbi = c_dbi/total;
        c_dbc = c_dbc/total;
        c_dbo = c_dbo/total;
        c_dby = c_dby/total;
        
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
