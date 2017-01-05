function [ l ] = DistributionPoissonLog( X, Theta, Beta )
    %
    % Calculate the log likelihood with the Poisson distribution (X ~ A*B)
    %
    [m, n] = size(X);
    if size(Theta, 1) ~= m
        fprintf('Dimension of Theta is wrong.');
        return
    end
    if size(Beta, 2) ~= n
        fprintf('Dimension of Beta is wrong.');
        return
    end
    if size(Theta, 2) ~= size(Beta, 1)
        fprintf('Dimension of latent parameter is different.');
        return
    end
    
    
    l = 0;
    
    cap_x = log(Theta * Beta);
    
    [x_X, y_X, v_X] = find(X);

    l = l - sum(Theta, 1) * sum(Beta, 2);
    
    for i = 1:length(v_X)
        a = v_X(i, 1) * cap_x(x_X(i,1), y_X(i,1)) - gammaln(v_X(i, 1) + 1);
        l = l + a;
    end
    
    
%     vecT = zeros(length(x_X),1);
%     for i = 1:length(v_X)
%         a = Theta(x_X(i,1),:);
%         b = Beta(:, y_X(i,1));
%         vecT(i,1) = Theta(x_X(i,1),:) * Beta(:, y_X(i,1));
%     end
%     l = l + sum(v_X .* log(vecT));
end

