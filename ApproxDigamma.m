function [ matX ] = ApproxDigamma( d, matC, itr )
    % Approximate the following equation: 
    % psi(matX) + d * matX = matC
    
    matX = matC;

    for t = 1:itr
        matX = matX - 0.5 * (psi(matX) + d .* matX - matC) .* (psi(1, matX) + d) ./ d;
        
        matX = matX .* (matX > 0);
        %fprintf('%f, %f \n', matX, diff);
        %fprintf('%f \n', (psi(matX) + d * matX - matC)^2);
    end
end

