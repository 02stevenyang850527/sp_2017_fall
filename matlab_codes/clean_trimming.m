function seq = clean_trimming(seq)
    k = 2;
    while (k < size(seq,2))
        if seq(k) ~= seq(k-1) && seq(k) ~= seq(k+1)
            seq(k)=[];
        end
        k = k+1;
    end
end