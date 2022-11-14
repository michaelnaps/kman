function [fig] = plot_comparisons(x1_list, x2_list, x0, tspan)

    [Nx, Ns] = size(x0);

    k = 0;

    size(x1_list)
    size(x2_list)
    size(tspan)

    fig = figure;
    for i = 1:Nx

        for j = 1:Ns

            subplot(Nx,Ns,k+j)
            hold on
            plot(tspan, x1_list(:,k+j), 'linewidth', 2)
            plot(tspan, x2_list(:,k+j), '--', 'linewidth', 1.5)
            legend('Model Func.', 'Koopman op.')
            title("x(" + i + "," + j + ")")
            hold off
            

        end

        k = k + Ns;

    end

end