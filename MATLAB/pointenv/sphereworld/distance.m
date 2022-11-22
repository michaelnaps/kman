function [dist] = distance(sphere, points)

    Np = length(points(:,1));

    if sphere.radius > 0
        adj = 1;
    else
        adj = -1;
    end
    
    dist = NaN(Np,1);
    for i = 1:Np
        
        dist(i) = norm(points(:,i) - sphere.xCenter);
        dist(i) = adj*dist(i) - adj*sphere.radius;

    end

end