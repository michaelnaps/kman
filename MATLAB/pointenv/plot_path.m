function [fig] = plot_path(world, robot, xGoal, xList, xComp)

    if nargin < 5
        xComp = [];
    end
  

    fig = figure;
    hold on

    if ~isempty(xList)
        robot.xCenter = xList(1,:)';
        plot(xList(:,1), xList(:,2), 'linewidth', 2);
    end

    plot_sphereworld(world, xGoal', fig);
    plot_sphere(robot, robot.color);

    if ~isempty(xComp)
        plot(xComp(:,1), xList(:,2), '--');
    end

    hold off

end