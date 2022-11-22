function [dist] = distance(sphere, points)

    Np = length(points(:,1));

    if sphere.radius > 0
        neg = 1;
    else
        neg = -1;
    end
    
    dist = NaN(Np,1);
    for i = 1:Np
        
        dist(i) = norm(points(:,i) - sphere.xCenter);
        dist(i) = neg*dist(i) - neg*sphere.radius;

    end

end