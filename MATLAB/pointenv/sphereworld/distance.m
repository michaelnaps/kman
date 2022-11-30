function [dist] = distance(sphere, point)

    if sphere.radius > 0
        neg = 1;
    else
        neg = -1;
    end
    
    dist = norm(point - sphere.xCenter');
    dist = neg*dist - sphere.radius;

end