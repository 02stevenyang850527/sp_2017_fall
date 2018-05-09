function [output_loss,output_acc,model] = solver(X_train,Y_train,alpha,batch_size,n_iter,...
                                                            Wf,Whf,Wi,Whi,Wc,Whc,Wo,Who,Wy,bf,bi,bc,bo,by)
    beta1 = 0.9;
    beta2 = 0.999;
    print_after = 1;
    H = 256;
    D = size(X_train,2);  % number of MFCC features
    output_dim = 61;
    
    cnt = 0;  % count for batch_size 
    m = size(X_train,3);
    smooth_loss = -log(1 / D);
    output_loss = zeros(1,floor(n_iter/print_after));
    output_acc = zeros(1,floor(n_iter/print_after));
    num_batch = ceil(m/batch_size);
    
   % M_Wf = zeros(D,H);
   % M_Wi = zeros(D,H);
   % M_Wc = zeros(D,H);
   % M_Wo = zeros(D,H);
   % M_Whf = zeros(1,H);
   % M_Whi = zeros(1,H);
   % M_Whc = zeros(1,H);
   % M_Who = zeros(1,H);
   % M_Wy = zeros(H,output_dim);
   % M_bf = zeros(1,H);
   % M_bi = zeros(1,H);
   % M_bc = zeros(1,H);
   % M_bo = zeros(1,H);
   % M_by = zeros(1,output_dim);
   %     
   % R_Wf = zeros(D,H);
   % R_Wi = zeros(D,H);
   % R_Wc = zeros(D,H);
   % R_Wo = zeros(D,H);
   % R_Whf = zeros(1,H);
   % R_Whi = zeros(1,H);
   % R_Whc = zeros(1,H);
   % R_Who = zeros(1,H);
   % R_Wy = zeros(H,output_dim);
   % R_bf = zeros(1,H);
   % R_bi = zeros(1,H);
   % R_bc = zeros(1,H);
   % R_bo = zeros(1,H);
   % R_by = zeros(1,output_dim);

    for iter = 1:n_iter
        if (mod(iter,100)==0)
            alpha = alpha - 9.9000e-04;
        end
        t = iter;
        if cnt >= num_batch
            cnt = 0;
        end
        
        if cnt == num_batch-1
            X_mini = X_train(:,:,cnt*batch_size+1:end);
            Y_mini = Y_train(cnt*batch_size+1:end,:);
        else
            X_mini = X_train(:,:,cnt*batch_size+1:(cnt+1)*batch_size);
            Y_mini = Y_train(cnt*batch_size+1:(cnt+1)*batch_size,:);
        end
        
        cnt = cnt + 1;
        
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
        c_loss = 0;
        total = 0;
        c_acc = 0;
        for batch_cnt = 1:size(X_mini,3)
            X = X_mini(:,:,batch_cnt);
            Y = Y_mini(batch_cnt,:);
            X = X(1:Y(end,end),:);
            Y = Y(1:Y(end));
            state_h = zeros(1,H);
            state_c = zeros(1,H);
            total = total + size(X,1);
   
            [dWf,dWhf,dWi,dWhi,dWc,dWhc,dWo,dWho,dWy,dbf,dbi,dbc,dbo,dby,loss,acc,~,~,record]...
            = train_step(X,Y,Wf,Whf,Wi,Whi,Wc,Whc,Wo,Who,Wy,bf,bi,bc,bo,by,state_h,state_c);
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
            c_loss = c_loss + loss;
            c_acc = c_acc + acc;
        end
        c_loss = c_loss/total;  % average loss of 1 frame-based sequence input
        c_acc = c_acc / total;

        smooth_loss = 0.999 .* smooth_loss + 0.001 .* c_loss;
        if mod(iter, print_after) == 0

            output_loss(iter/print_after) = c_loss;
            output_acc(iter/print_after) = c_acc;

            disp('======================================');
            fprintf('Iter: %d\n',int64(iter));
            fprintf('train_loss: %f, train_acc: %f\n',c_loss, c_acc);
            disp('======================================');
            disp(' ');
        end

%         c_dWf = round(c_dWf/total, 3);
%         c_dWi = round(c_dWi/total, 3);
%         c_dWc = round(c_dWc/total, 3);
%         c_dWo = round(c_dWo/total, 3);
%         c_dWhf = round(c_dWhf/total, 3);
%         c_dWhi = round(c_dWhi/total, 3);
%         c_dWhc = round(c_dWhc/total, 3);
%         c_dWho = round(c_dWho/total, 3);
%         c_dWy = round(c_dWy/total, 3);
%         c_dbf = round(c_dbf/total, 3);
%         c_dbi = round(c_dbi/total, 3);
%         c_dbc = round(c_dbc/total, 3);
%         c_dbo = round(c_dbo/total, 3);
%         c_dby = round(c_dby/total, 3);

        c_dWf = c_dWf/total;
        c_dWi = c_dWi/total;
        c_dWc = c_dWc/total;
        c_dWo = c_dWo/total;
        c_dWhf = c_dWhf/total;
        c_dWhi = c_dWhi/total;
        c_dWhc = c_dWhc/total;
        c_dWho = c_dWho/total;
        c_dWy = c_dWy/total;
        c_dbf = c_dbf/total;
        c_dbi = c_dbi/total;
        c_dbc = c_dbc/total;
        c_dbo = c_dbo/total;
        c_dby = c_dby/total;
        
       % M_Wf = exp_running_avg(M_Wf, c_dWf, beta1);
       % M_Wi = exp_running_avg(M_Wi, c_dWi, beta1);
       % M_Wc = exp_running_avg(M_Wc, c_dWc, beta1);
       % M_Wo = exp_running_avg(M_Wo, c_dWo, beta1);
       % M_Whf = exp_running_avg(M_Whf, c_dWhf, beta1);
       % M_Whi = exp_running_avg(M_Whi, c_dWhi, beta1);
       % M_Whc = exp_running_avg(M_Whc, c_dWhc, beta1);
       % M_Who = exp_running_avg(M_Who, c_dWho, beta1);
       % M_Wy = exp_running_avg(M_Wy, c_dWy, beta1);
       % M_bf = exp_running_avg(M_bf, c_dbf, beta1);
       % M_bi = exp_running_avg(M_bi, c_dbi, beta1);
       % M_bc = exp_running_avg(M_bc, c_dbc, beta1);
       % M_bo = exp_running_avg(M_bo, c_dbo, beta1);
       % M_by = exp_running_avg(M_by, c_dby, beta1);
    
       % R_Wf = exp_running_avg(R_Wf, c_dWf.^2, beta2);
       % R_Wi = exp_running_avg(R_Wi, c_dWi.^2, beta2);
       % R_Wc = exp_running_avg(R_Wc, c_dWc.^2, beta2);
       % R_Wo = exp_running_avg(R_Wo, c_dWo.^2, beta2);
       % R_Whf = exp_running_avg(R_Whf, c_dWhf.^2, beta2);
       % R_Whi = exp_running_avg(R_Whi, c_dWhi.^2, beta2);
       % R_Whc = exp_running_avg(R_Whc, c_dWhc.^2, beta2);
       % R_Who = exp_running_avg(R_Who, c_dWho.^2, beta2);
       % R_Wy = exp_running_avg(R_Wy, c_dWy.^2, beta2);
       % R_bf = exp_running_avg(R_bf, c_dbf.^2, beta2);
       % R_bi = exp_running_avg(R_bi, c_dbi.^2, beta2);
       % R_bc = exp_running_avg(R_bc, c_dbc.^2, beta2);
       % R_bo = exp_running_avg(R_bo, c_dbo.^2, beta2);
       % R_by = exp_running_avg(R_by, c_dby.^2, beta2);
       % 
       % m_Wf_hat = M_Wf ./ (1 - beta1^t);
       % m_Wi_hat = M_Wi ./ (1 - beta1^t);
       % m_Wc_hat = M_Wc ./ (1 - beta1^t);
       % m_Wo_hat = M_Wo ./ (1 - beta1^t);
       % m_Whf_hat = M_Whf ./ (1 - beta1^t);
       % m_Whi_hat = M_Whi ./ (1 - beta1^t);
       % m_Whc_hat = M_Whc ./ (1 - beta1^t);
       % m_Who_hat = M_Who ./ (1 - beta1^t);
       % m_Wy_hat = M_Wy ./ (1 - beta1^t);
       % m_bf_hat = M_bf ./ (1 - beta1^t);
       % m_bi_hat = M_bi ./ (1 - beta1^t);
       % m_bc_hat = M_bc ./ (1 - beta1^t);
       % m_bo_hat = M_bo ./ (1 - beta1^t);
       % m_by_hat = M_by ./ (1 - beta1^t);
       % 
       % r_Wf_hat = R_Wf ./ (1 - beta2^t);
       % r_Wi_hat = R_Wi ./ (1 - beta2^t);
       % r_Wc_hat = R_Wc ./ (1 - beta2^t);
       % r_Wo_hat = R_Wo ./ (1 - beta2^t);
       % r_Whf_hat = R_Whf ./ (1 - beta2^t);
       % r_Whi_hat = R_Whi ./ (1 - beta2^t);
       % r_Whc_hat = R_Whc ./ (1 - beta2^t);
       % r_Who_hat = R_Who ./ (1 - beta2^t);
       % r_Wy_hat = R_Wy ./ (1 - beta2^t);
       % r_bf_hat = R_bf ./ (1 - beta2^t);
       % r_bi_hat = R_bi ./ (1 - beta2^t);
       % r_bc_hat = R_bc ./ (1 - beta2^t);
       % r_bo_hat = R_bo ./ (1 - beta2^t);
       % r_by_hat = R_by ./ (1 - beta2^t);
       % 
       % Wf = Wf - alpha .* m_Wf_hat ./ (sqrt(r_Wf_hat) + 1e-8);
       % Wi = Wi - alpha .* m_Wi_hat ./ (sqrt(r_Wi_hat) + 1e-8);
       % Wc = Wc - alpha .* m_Wc_hat ./ (sqrt(r_Wc_hat) + 1e-8);
       % Wo = Wo - alpha .* m_Wo_hat ./ (sqrt(r_Wo_hat) + 1e-8);
       % Whf = Whf - alpha .* m_Whf_hat ./ (sqrt(r_Whf_hat) + 1e-8);
       % Whi = Whi - alpha .* m_Whi_hat ./ (sqrt(r_Whi_hat) + 1e-8);
       % Whc = Whc - alpha .* m_Whc_hat ./ (sqrt(r_Whc_hat) + 1e-8);
       % Who = Who - alpha .* m_Who_hat ./ (sqrt(r_Who_hat) + 1e-8);
       % Wy = Wy - alpha .* m_Wy_hat ./ (sqrt(r_Wy_hat) + 1e-8);
       % bf = bf - alpha .* m_bf_hat ./ (sqrt(r_bf_hat) + 1e-8);
       % bi = bi - alpha .* m_bi_hat ./ (sqrt(r_bi_hat) + 1e-8);
       % bc = bc - alpha .* m_bc_hat ./ (sqrt(r_bc_hat) + 1e-8);
       % bo = bo - alpha .* m_bo_hat ./ (sqrt(r_bo_hat) + 1e-8);
       % by = by - alpha .* m_by_hat ./ (sqrt(r_by_hat) + 1e-8);

%        Wf = round(Wf - alpha .* c_dWf, 3);
%        Wi = round(Wi - alpha .* c_dWi, 3);
%        Wc = round(Wc - alpha .* c_dWc, 3);
%        Wo = round(Wo - alpha .* c_dWo, 3);
%        Whf = round(Whf - alpha .* c_dWhf, 3);
%        Whi = round(Whi - alpha .* c_dWhi, 3);
%        Whc = round(Whc - alpha .* c_dWhc, 3);
%        Who = round(Who - alpha .* c_dWho, 3);
%        bf = round(bf - alpha .* c_dbf, 3);
%        bi = round(bi - alpha .* c_dbi, 3);
%        bc = round(bc - alpha .* c_dbc, 3);
%        bo = round(bo - alpha .* c_dbo, 3);
%        Wy = round(Wy - alpha .* c_dWy, 3);
%        by = round(by - alpha .* c_dby, 3);
       
       Wf = Wf - alpha .* c_dWf;
       Wi = Wi - alpha .* c_dWi;
       Wc = Wc - alpha .* c_dWc;
       Wo = Wo - alpha .* c_dWo;
       Whf = Whf - alpha .* c_dWhf;
       Whi = Whi - alpha .* c_dWhi;
       Whc = Whc - alpha .* c_dWhc;
       Who = Who - alpha .* c_dWho;
       bf = bf - alpha .* c_dbf;
       bi = bi - alpha .* c_dbi;
       bc = bc - alpha .* c_dbc;
       bo = bo - alpha .* c_dbo;
       Wy = Wy - alpha .* c_dWy;
       by = by - alpha .* c_dby;
       
    end
    
    model.Wf = Wf;
    model.Wi = Wi;
    model.Wc = Wc;
    model.Wo = Wo;
    model.Whf = Whf;
    model.Whi = Whi;
    model.Whc = Whc;
    model.Who = Who;
    model.Wy = Wy;
    model.bf = bf;
    model.bi = bi;
    model.bc = bc;
    model.bo = bo;
    model.by = by;
    
end
