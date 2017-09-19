function char2idx = encoder(word,txt)
    [~,m] = size(txt);
    char2idx = zeros(1,m);
    for k = 1:m
        char2idx(k)= strfind(word,txt(k));
    end
end