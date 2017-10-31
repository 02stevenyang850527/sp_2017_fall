x_f = matfile('x.mat');
y_f = matfile('y.mat');
x=x_f.x;
y=y_f.y;

maxlen = 0;
for k = 1:size(x,2)
    if size(x{k},1) > maxlen
        maxlen = size(x{k},1);
    end
end

maxlen = maxlen + 3; % maxlen = 777 + 3

x_pad = zeros(maxlen,39,size(x,2));
y_pad = zeros(size(x,2),maxlen);

for k=1:size(x,2)
    len_mark = size(y{k},2);
    x_pad(:,:,k) = [x{k};zeros(maxlen-len_mark, 39)];
    y_pad(k,:) = [y{k},zeros(1,maxlen-len_mark)];
    y_pad(k,end) = len_mark;
end
