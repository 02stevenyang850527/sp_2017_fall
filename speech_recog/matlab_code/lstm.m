
rng(10);
xfile = matfile('x_pad.mat');
yfile = matfile('y_pad.mat');
x = xfile.x_pad;
y = yfile.y_pad;
index = randperm(length(y));
% shffule training data
x = x(:,:,index);
y = y(index,:);

% 5 percent of validation data
x_train = x(:,:,index(1:end-185));
y_train = y(index(1:end-185),:);
x_valid = x(:,:,index(end-184:end));
y_valid = y(index(end-184:end),:);

H = 256;
alpha = 1e-3;
batch_size = 8;
n_iter = 3000;

D = size(x,2);
output_dim = 48;
Z = H + D;

Wf = randn(Z,H) ./ sqrt(Z/2);
Wi = randn(Z,H) ./ sqrt(Z/2);
Wc = randn(Z,H) ./ sqrt(Z/2);
Wo = randn(Z,H) ./ sqrt(Z/2);
Wy = randn(H,output_dim) ./ sqrt(output_dim/2);

bf = ones(1,H);
bi = zeros(1,H);
bc = zeros(1,H);
bo = zeros(1,H);
by = zeros(1,output_dim);


[train_loss,train_acc,valid_loss,valid_acc] ...
= solver(x_train,y_train,x_valid,y_valid,alpha,batch_size,n_iter,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by);

disp(valid_acc);
disp(valid_loss);

save('train_loss.mat','train_loss');
save('train_acc.mat','train_acc');
save('valid_loss.mat','valid_loss');
save('valid_acc.mat','valid_acc');

model.Wf = Wf;
model.Wi = Wi;
model.Wc = Wc;
model.Wo = Wo;
model.Wy = Wy;
model.bf = bf;
model.bi = bi;
model.bc = bc;
model.bo = bo;
model.by = by;

save('model.mat','-struct','model');

exit

