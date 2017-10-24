function res = l2_reg(W, lam)
    if nargin < 2
        lam = 1e-3;
    end
    res = 0.5 * lam * sum(W .* W);
end