function [ l ] = DistributionPoissonLogNZ( X, XX )
    %
    % Calculate the log likelihood with the Poisson distribution (X ~ A*B)
    %       
    l = 0;
    
    cap_x = log(XX);
    
    [x_X, y_X, v_X] = find(X);
    listtt = [];
    for i = 1:length(v_X)
        a = v_X(i, 1) * cap_x(x_X(i,1), y_X(i,1)) - gammaln(v_X(i, 1) + 1) - XX(x_X(i,1), y_X(i,1));
        l = l + a;
        listtt = [listtt;a];
    end
end
