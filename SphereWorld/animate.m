function [fig] = animate(world, robot, xGoal, tspan, xList, xComp, dList)

    if nargin < 7
        dList = [];
    else
        Nw = size(dList, 2);
    end

    if nargin < 6
        xComp = [];
    end
    
    N = length(tspan);
    if length(tspan) > 1
        dt = tspan(2) - tspan(1);
    else
        dt = 1;
    end

    x1 = xList(:,1);
    x2 = xList(:,2);
    
    fig = figure;
    for i = 1:N
        clf(fig);

        plot_sphereworld(world, xGoal', fig);

        robot.x = [x1(i); x2(i)];
        plot_sphere(robot, robot.color);

        if ~isempty(xComp)
            scatter(xComp(i,1), xComp(i,2), 'g*');
        end

        if ~isempty(dList)
            for j = 1:Nw
                obsDistance = struct;
                obsDistance.x = robot.x;
                obsDistance.r = -sqrt(dList(i,j));
                obsDistance.color = [0.4660 0.6740 0.1880];

                plot_sphere(obsDistance, obsDistance.color);
            end
        end

        ylim([world(1).r, -world(1).r])
        xlim([world(1).r, -world(1).r])

        pause(dt);
    end
end