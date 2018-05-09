function res = trimming(seq)
    len = size(seq,2);
    res = seq(1);
    
    for k=2:len
        if (seq(k) ~= seq(k-1))
            res = [res,seq(k)];
        end
    end
end