%function sphereworld_plot(world,xGoal)
%Uses sphere_draw to draw the spherical obstacles together with a  * marker at
%the goal location.
function plot_sphereworld(world, xGoal)
if nargin < 2
    xGoal = [];
end

nbSpheres=size(world,2);
for iSphere=1:nbSpheres
    plot_sphere(world(iSphere),'b');
    hold on
end

if ~isempty(xGoal)
    if exist('xGoal','var')
        plot(xGoal(1,:),xGoal(2,:),'r*');
    end
end

axis equal
