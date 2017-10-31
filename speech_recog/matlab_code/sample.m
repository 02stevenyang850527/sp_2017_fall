function [loss, acc] = sample(X_valid,Y_valid,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,acc,loss)
    num_valid = size(X_valid,2);
    H = 256;
    total = 0;
    for n = 1:num_valid
        X = X_valid{n};
        Y = Y_valid{n};
        state_h = zeros(1,H);
        state_c = zeros(1,H);
        sentence_length = size(X,1);
        total = total + sentence_length;
        for k = 1:sentence_length
            [~,~,~,~,~,~,~,~,~,~,state_c,Wy,Y_pred,state_h] ...
            = forward(Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by,X(k,:),state_h,state_c,false);
            loss = loss + cross_entropy(Y_pred, Y(k), 0, Wf, Wi, Wc, Wo, Wy);
            [~,pred] = max(Y_pred);
            if (pred == Y(k))
                acc = acc +1;
            end
        end
    end
    acc = acc/total;
    loss = loss/total;
end