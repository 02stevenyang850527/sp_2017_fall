rng(10);

x1file = matfile('x_test_full.mat');
y1file = matfile('y_test_full.mat');

x_test = x1file.x;
y_test = y1file.y;

model = matfile('model.mat');
H = 256;
my_y ={};

for k=1:size(x_test, 3)
    disp(k)
    X = x_test(:,:,k);
    Y = y_test(k,:);
    sentence_len = Y(end);
    X = X(1:sentence_len, :);
    Y = Y(1:sentence_len);
    temp = zeros(1,sentence_len);

    state_h = zeros(1,H);
    state_c = zeros(1,H);
    for s=1:sentence_len
        [~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,state_c,Wy,Y_pred,state_h] ...
        = forward(model.Wf,model.Whf,model.Wi,model.Whi,model.Wc,model.Whc,model.Wo,model.Who,...
            model.Wy,model.bf,model.bi,model.bc,model.bo,model.by,X(s,:),state_h,state_c,false);
        [~,pred] = max(Y_pred);
        temp(s) = pred;
    end
    my_y{k} = temp;
end

save('y_pred_full.mat','my_y')

exit
