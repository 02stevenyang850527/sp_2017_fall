function res = exp_running_avg(running, new, gamma)
    res = gamma .* running + (1 - gamma) .* new;
end