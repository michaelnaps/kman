function [dist] = distance(sphere, point)

    if sphere.radius > 0
        neg = 1;
    else
        neg = -1;
    end
        
    dist = sqrt(sum((point - sphere.xCenter').^2));
    dist = neg*dist - sphere.radius;

end