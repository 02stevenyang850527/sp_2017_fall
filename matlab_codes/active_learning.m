x1file = matfile('x_test_full.mat');
y1file = matfile('y_test_full.mat');

x_test = x1file.x;
y_test = y1file.y;

model = matfile('./standard_model/model.mat');
H = 256;
my_y ={};

for k=1:size(x_test, 3)
    
    X = x_test(:,:,k);
    Y = y_test(k,:);
    sentence_len = Y(end);
    X = X(1:sentence_len, :);
    Y = Y(1:sentence_len);
    
    state_h = zeros(1,H);
    state_c = zeros(1,H);
    
    [model, loss, acc, state_h, state_c, record]...
    = test_step_active(X_test,y_test,model,state_h,state_c);

    disp('======================================');
    fprintf('No. %d sentence\n',int64(k));
    fprintf('train_loss: %f, train_acc: %f\n',loss, acc);
    disp('======================================');
    disp(' ');
    my_y{k} = record;
end

save('model_active_1.mat','-struct','model');
save('y_pred_full_0.mat','my_y')

