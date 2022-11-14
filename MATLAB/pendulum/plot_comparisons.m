function [fig] = plot_comparisons(x1_list, x2_list, x0, tspan)

    Nx = length(x0(:,1));
    Ns = length(x0(1,:));

    k = 0;

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

        k = k + j;

    end

end