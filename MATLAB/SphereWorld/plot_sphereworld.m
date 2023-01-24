%function sphereworld_plot(world,xGoal)
%Uses sphere_draw to draw the spherical obstacles together with a  * marker at
%the goal location.
function [fig] = plot_sphereworld(world, xGoal, fig)

if nargin < 3
    fig = figure;
end

if nargin < 2
    xGoal = [];
end

nbSpheres=size(world,2);
for iSphere=1:nbSpheres
    plot_sphere(world(iSphere), 'k');
    hold on
    if world(iSphere).r > 0
        plot(world(iSphere).x(1), world(iSphere).x(2), 'kx')
    end
end

if ~isempty(xGoal)
    if exist('xGoal','var')
        plot(xGoal(1,:),xGoal(2,:),...
            'marker', 'square',...
            'color', [0.4660 0.6740 0.1880],...
            'linewidth', 2,...
            'markersize', 8);
    end
end

axis equal

end
