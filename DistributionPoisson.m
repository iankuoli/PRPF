function [ p ] = DistributionPoisson( x, lambda )
    p = exp(-lambda);
    p = p .* (lambda .^ x);
    p = p ./ gamma(x+1);
end

