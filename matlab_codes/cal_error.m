f1 = matfile('y_valid.mat');
y_valid = f1.y_valid;

% 48-to-39 phone conversion
y_valid(y_valid== 4) =  1;
y_valid(y_valid== 6) =  3;
y_valid(y_valid==10) = 38;
y_valid(y_valid==15) = 28;
y_valid(y_valid==16) = 30;
y_valid(y_valid==17) = 38;
y_valid(y_valid==24) = 23;
y_valid(y_valid==44) = 38;
y_valid(y_valid==48) = 37;

f2 = matfile('my_y.mat');
y_pred = f2.my_y;


phone_acc = zeros(1,185);
total = 0;
frame_acc = 0;

for k=1:185
    valid = y_valid(k,:);
    total = total + valid(end);
    valid = valid(1:valid(end));
    
    pred = y_pred{k};
    
    % 48-to-39 phone conversion
    pred(pred== 4) =  1;
    pred(pred== 6) =  3;
    pred(pred==10) = 38;
    pred(pred==15) = 28;
    pred(pred==16) = 30;
    pred(pred==17) = 38;
    pred(pred==24) = 23;
    pred(pred==44) = 38;
    pred(pred==48) = 37;
    
    frame_acc = frame_acc + sum(pred == valid);
    
    valid = trimming(valid);
    pred = clean_trimming(pred);
    pred = trimming2(pred);
    [score,~] = EditDistance(pred,valid);
    %score = strdist(pred, valid);
    phone_acc(k) = (size(valid,2) - score) / size(valid,2);
end
frame_acc = frame_acc/total;