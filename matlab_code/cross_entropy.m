function res = cross_entropy(y_pred, y_train, lam, Wf, Wi, Wc, Wo, Wy)
    % Notes:
    % 1. y_pred is in one hot format, in this case: size = 1x71
    % 2. y_train is an index, in this case: from 1~71
    
    prob = my_softmax(y_pred);
    log_like = -log(prob(y_train));
    data_loss = sum(log_like);
    loss_f = l2_reg(Wf, lam);
    loss_i = l2_reg(Wi, lam);
    loss_c = l2_reg(Wc, lam);
    loss_o = l2_reg(Wo, lam);
    loss_y = l2_reg(Wy, lam);
    reg_loss = sum(loss_f) + sum(loss_i) + sum(loss_c) + sum(loss_o) + sum(loss_y);
    res = data_loss + reg_loss;
    
end