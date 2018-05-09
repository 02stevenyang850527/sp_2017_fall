function prob = my_softmax(X)
    % X is a 1-dim array, that is 1 x m

%     precision = 15;      
%     X = ceil(X-max(X));   % Shift input X and truncate the decimal part
%                           
%     eX = 2.^X;            % Use 2-base exponential. (X is integer after truncation)
% 
%     eX(X< -precision) = 0; % Precision depends on design.
%                            % In this case, we negelect exponential values < 2^(-15)
% 
%     prob = eX./sum(eX);   % Normalize exponential terms
    eX = exp(X - max(X));
    prob = eX./sum(eX);
end
