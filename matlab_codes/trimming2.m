function res = trimming2(seq)
    res= [];
    k = 1;
    cnt = 0;
    while (k < size(seq,2))
        if (seq(k) == seq(k+1))
            cnt = cnt +1;
        else
            if (cnt > 1)
                res = [res,seq(k-cnt)];
            end
            cnt = 0;
        end
        k = k+1;
    end
end