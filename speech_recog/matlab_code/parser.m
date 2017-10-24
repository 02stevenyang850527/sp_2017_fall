
label_dir = './train_aligned.lab';
mfcc_dir = './train.ark';
map_dir = './48phone_char.map';

[wav_name,label] = textread(label_dir,'%s%s','whitespace',',');
mfcc = importdata(mfcc_dir);
[phone,index,symbol] = textread(map_dir,'%s%d%s');

mark=[];
y_index = zeros(1,length(label));          % for training
% parse the data

for k = 1:length(label)-1
    x = strfind(wav_name{k},'_');
    for m = 1:length(phone)
        if isequal(label{k},phone{m})
            y_index(k) = index(m);
            break;
        end
    end
    if (length(wav_name{k+1}) < x(2))
        mark = [mark,k];
    elseif sum(wav_name{k}(1:x(2)) ~= wav_name{k+1}(1:x(2)))
        mark = [mark,k];
    end
end
y_index(length(label)) = 38;

y = cell(1,length(mark)+1);
x = cell(1,length(mark)+1);
pre = 0;
for k =1:length(mark)
    pre = pre + 1;
    y{k} = y_index(pre:mark(k));
    x{k} = mfcc.data(pre:mark(k),:);
    pre = mark(k);
end

y{length(mark)+1} = y_index(pre+1:end);
x{length(mark)+1} = mfcc.data(pre+1:end,:);

save('y.mat','y');
save('x.mat','x');
disp('parsing exit...')
exit

