function result = one_hot(word,char2idx)
    [~,m] = size(word);
    [~,n] = size(char2idx);
    result = zeros(n,m);
    for k = 1:n
        result(k,char2idx(k))=1;
    end
end