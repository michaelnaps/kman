function [fig] = plot_path(world, robot, xGoal, xList, xComp)

    if nargin < 5
        xComp = [];
    end
  

    fig = figure;
    hold on

    plot_sphereworld(world, xGoal', fig);

    if ~isempty(xList)
        robot.xCenter = xList(1,:)';
        plot(xList(:,1), xList(:,2), 'linewidth', 2);
    else
        disp("x-List is empty...")
    end

    if ~isempty(xComp)
        plot(xComp(:,1), xList(:,2), '--');
    end

    plot_sphere(robot, robot.color);

    hold off

end