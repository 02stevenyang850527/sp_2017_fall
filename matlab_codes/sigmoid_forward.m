function out = sigmoid_forward(X)
    %out = 1 ./ (1+exp(-X));
    out = zeros(size(X));
    for k=1:size(out,2)
        if -1 < X(k) && X(k) <= 1
            out(k) = 0.5 + 0.25*X(k);
        elseif 1 < X(k) && X(k) <= 2
            out(k) = 0.609375 + 0.140625*X(k);
        elseif 2 < X(k) && X(k) <= 3
            out(k) = 0.671875 + 0.109375*X(k);
        elseif 3 < X(k)
            out(k) = 1;
        elseif -2 < X(k) && X(k) <= -1
            out(k) = 0.390625 + 0.140625*X(k);
        elseif -3 < X(k) && X(k) <= -2
            out(k) = 0.328125 + 0.109375*X(k);
        else
            out(k) = 0;
        end
    end

end
