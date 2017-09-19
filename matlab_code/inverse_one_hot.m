function id = inverse_one_hot(X)
    % size(X) = 1xn
    [~,n] = size(X);
    for k=1:n
        if X(k)==1
            id = k;
            break;
        end
    end
end