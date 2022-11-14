function [] = animate(t, th1_list, th2_list, names, adj, fignum)

    [th2_list, names, adj, f] = check_argin...
    (th1_list, th2_list, names, adj, fignum);

    dt = t(2) - t(1);
    th1 = th1_list(:,1);
    th2 = th2_list(:,1);

    pivot = [0, 0];
    x1 = cos(th1);
    y1 = sin(th1);
    
    x2 = cos(th2);
    y2 = sin(th2);

    figure(f);
    f.Position = [0, 0, 300, 300];

    for i = 1:adj:length(t)

        f = clf(f);

        % th1
        plot([pivot(1), x1(i)], [pivot(2), y1(i)], 'k', 'linewidth', 2); hold on
        plot(x1(i), y1(i), 'ro', 'markersize', 8); hold on

        % th2
        if nargin > 2
            plot([pivot(1), x2(i)], [pivot(2), y2(i)], 'k--', 'linewidth', 2); hold on
            plot(x2(i), y2(i), 'b*', 'markersize', 8); hold on
        end

        xlim([-1, 1]);
        ylim([-1, 1]);

        legend(names);
        pause(adj*dt);

    end
end

function [th2_list, names, adj, f] = check_argin...
         (th1_list, th2_list, names, adj, fignum)
    
    if nargin < 6
        f = figure;
    else
        f = figure(fignum);
    end

    if nargin < 5
        adj = 10;
    end

    if nargin < 4
        names = ["", "th1", "", "th2"];
    end

    if nargin < 3
        th2_list = th1_list;
        names = ["", "th1"];
    end
    
end