function [fig] = plot_comparisons(x1_list, x2_list, x0, tspan)

    [Nx, Ns] = size(x0);

    k = 0;

    fig = figure;
    for i = 1:Nx

        for j = 1:Ns

            subplot(Nx,Ns,k+j)
            hold on
            plot(tspan, x1_list(:,k+j), 'linewidth', 2)
            plot(tspan, x2_list(:,k+j), '--', 'linewidth', 1.5)
            title("x(" + i + "," + j + ")")
            
            if i == 1 && j == Ns
                legend('Model Func.', 'Koopman op.')
            end

            ylim([min(x1_list(:,k+j))-1, max(x1_list(:,k+j))+1])
            hold off
            

        end

        k = k + Ns;

    end

end