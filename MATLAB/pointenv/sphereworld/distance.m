function [dist] = distance(sphere, points)

    Np = length(points(:,1));
    
    dist = NaN(Np,1);
    for i = 1:Np
        
        dist(i) = norm(points(:,i) - sphere.center);
        dist(i) = dist(i) - sphere.radius;

    end

end