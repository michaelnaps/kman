function [grad] = distance_grad(sphere, point, dist)

    if nargin < 3
        dist = distance(sphere, point);
    end

    if dist > 0
        grad = (point - sphere.xCenter') / dist;
    else
        grad = [0, 0];
    end

end