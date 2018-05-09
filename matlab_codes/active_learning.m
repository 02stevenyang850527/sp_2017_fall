x1file = matfile('x_test_full.mat');
y1file = matfile('y_test_full.mat');

x_test = x1file.x;
y_test = y1file.y;

model = matfile('model.mat');
H = 256;
my_y ={};

for k=1:size(x_test, 3)
    
    X = x_test(:,:,k);
    Y = y_test(k,:);
    sentence_len = Y(end);
    X = X(1:sentence_len, :);
    Y = Y(1:sentence_len);
    temp = zeros(1,sentence_len);
    
    disp('======================================');
    fprintf('No. %d sentence\n',int64(k));
    fprintf('train_loss: %f, train_acc: %f\n',c_loss, c_acc);
    disp('======================================');
    disp(' ');
    
end