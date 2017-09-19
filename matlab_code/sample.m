function sample(X_seed,amount,idx2char,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,state_h,state_c) %idx2char=word
    X = inverse_one_hot(X_seed);
    chars = char(zeros(1,amount));
    chars(1) = idx2char(X);
    idx_list = 1:length(X_seed);
    
    for k = 1:amount-1
        [~,~,~,~,~,~,~,~,~,~,state_c,~,prob,state_h]...
         = forward(Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,X_seed,state_h,state_c,false);

        idx = datasample(idx_list,1,'Weights',prob);
        chars(k+1) = idx2char(idx);
        X_seed = zeros(1,length(idx2char));
        X_seed(idx) = 1;
    end
    disp(chars);
end