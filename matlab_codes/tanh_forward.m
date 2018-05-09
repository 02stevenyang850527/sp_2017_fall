function out = tanh_forward(X)
    %out = tanh(X);
    out = zeros(size(X));
    for k = 1:size(X,2)
        if -1 < X(k) && X(k) <= 1
            out(k) = 0.76171875*X(k);
        elseif 1 < X(k) && X(k) <= 2
            out(k) = 0.5625 + 0.19921875*X(k);
        elseif 2 < X(k) && X(k) <= 3
            out(k) = 0.890625 + 0.03515625*X(k);
        elseif 3 < X(k)
            out(k) = 1;
        elseif -2 < X(k) && X(k) <= -1
            out(k) = -0.5625 + 0.19921875*X(k);
        elseif -3 < X(k) && X(k) <= -2
            out(k) = -0.890625 + 0.03515625*X(k);
        else
            out(k) = -1;
        end
    end
end
