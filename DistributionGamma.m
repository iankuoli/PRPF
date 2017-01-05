function [ p ] = DistributionGamma( x, shp, rte )
    
    p = (rte .^ shp) ./ gamma(shp) .* (x .^ (shp - 1)) .* exp(-rte .* x);

end

