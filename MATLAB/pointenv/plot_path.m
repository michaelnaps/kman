function [fig] = plot_path(robot, xList, world, xGoal, xComp)

    if nargin < 5
        xComp = [];
    end

    robot.xCenter = xList(1,:)';

    fig = figure;
    hold on
    plot_sphereworld(world, xGoal', fig);
    plot_sphere(robot, robot.color);
    plot(xList(:,1), xList(:,2), 'linewidth', 2);

    if ~isempty(xComp)
        plot(xComp(:,1), xList(:,2), '--');
    end

    hold off

end