function [fig] = animate(robot, x_list, tspan, world, xGoal, x_comp)

    if nargin < 6
        x_comp = [];
    end

    fig = figure;
%     plot_sphereworld(world, xGoal);
    
    N = length(tspan);
    dt = tspan(2) - tspan(1);

    x1 = x_list(:,1);
    x2 = x_list(:,2);
    
    for i = 1:N
        clf(fig);

        plot_sphereworld(world, xGoal', fig);

        robot.xCenter = [x1(i); x2(i)];
        plot_sphere(robot, robot.color);

        if ~isempty(x_comp)
            scatter(x_comp(i,1), x_comp(i,2), 'g*');
        end

        pause(dt);
    end
end