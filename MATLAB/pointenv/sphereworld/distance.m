function [dist] = distance(sphere, point)

    if sphere.r > 0
        neg = 1;
    else
        neg = -1;
    end
        
    dist = norm(point - sphere.x);
    dist = neg*dist - sphere.r;

end