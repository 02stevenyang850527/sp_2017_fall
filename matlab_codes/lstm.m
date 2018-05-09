
rng(10);

xfile = matfile('x_train_full.mat');
yfile = matfile('y_train_full.mat');

x1file = matfile('x_test_full.mat');
y1file = matfile('y_test_full.mat');

x_train = xfile.x;
y_train = yfile.y;

x_test = x1file.x;
y_test = y1file.y;
index = randperm(length(y_train));

x_train = x_train(:,:,index);
y_train = y_train(index,:);



%xfile = matfile('x_pad.mat');
%yfile = matfile('y_pad.mat');
%x = xfile.x_pad;
%y = yfile.y_pad;
%index = randperm(length(y));
%shffule training data
%x = x(:,:,index);
%y = y(index,:);
%
% 5 percent of validation data
%x_train = x(:,:,index(1:end-185));
%y_train = y(index(1:end-185),:);
%x_valid = x(:,:,index(end-184:end));
%y_valid = y(index(end-184:end),:);


H = 256;
alpha = 1;
batch_size = 8;
n_iter = 100000; % around 173 epochs when batch=8

D = size(x_train,2);
output_dim = 61;

% Xavier Initialization
pd = makedist('Normal');
fan1_avg = (D+H)/2;
fan2_avg = (H+output_dim)/2;
scale1 = 1/max(1, fan1_avg);
scale2 = 1/max(1, fan2_avg);

std1 = sqrt(scale1);
std2 = sqrt(scale2);
t1 = truncate(pd,-2*std1,2*std1);
t2 = truncate(pd,-2*std2,2*std2);

Wf = random(t1,D,H); 
Wi = random(t1,D,H); 
Wc = random(t1,D,H); 
Wo = random(t1,D,H); 
Whf = random(t1,1,H); 
Whi = random(t1,1,H);
Whc = random(t1,1,H);
Who = random(t1,1,H);
Wy = random(t2,H,output_dim);

bf = ones(1,H);
bi = random(t1,1,H);
bc = random(t1,1,H);
bo = random(t1,1,H);
by = random(t2,1,output_dim);

disp('Start Training!')

[train_loss,train_acc, model] ...
= solver(x_train,y_train,alpha,batch_size,n_iter,Wf,Whf,Wi,Whi,Wc,Whc,Wo,Who,Wy,bf,bi,bc,bo,by);

save('train_loss.mat','train_loss');
save('train_acc.mat','train_acc');

%save('model_quantize_batch1.mat','-struct','model');
save('model.mat','-struct','model');

exit

