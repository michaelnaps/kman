function [grad] = distancegrad(sphere, points)

    [~, Np] = size(points);

    dist = distance(sphere, points);
    grad = NaN(2, Np);

    for i = 1:Np

        if dist(i) > 0
            grad(:,i) = (points(:,i) - sphere.xCenter) / dist(i);
        else
            grad(:,i) = [0; 0];
        end

    end

end