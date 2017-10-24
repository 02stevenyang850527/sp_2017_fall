
rng(10);
xfile = matfile('x.mat');
yfile = matfile('y.mat');
x = xfile.x;
y = yfile.y;
index = randperm(length(y));

H = 256;
alpha = 1e-3;
batch_size = 8;
n_iter = 4000;

[~,D] = size(x{1});
output_dim = 48;
Z = H + D;

Wf = randn(Z,H) ./ sqrt(Z/2);
Wi = randn(Z,H) ./ sqrt(Z/2);
Wc = randn(Z,H) ./ sqrt(Z/2);
Wo = randn(Z,H) ./ sqrt(Z/2);
Wy = randn(H,output_dim) ./ sqrt(output_dim/2);

bf = zeros(1,H);
bi = zeros(1,H);
bc = zeros(1,H);
bo = zeros(1,H);
by = zeros(1,output_dim);

x = x(index);
y = y(index);

[output_loss1,output_acc1] = solver(x,y,alpha,batch_size,n_iter,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by);

save('output_loss1.mat','output_loss1');
save('output_acc1.mat','output_acc1');
exit

