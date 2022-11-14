function [K, acc, ind, err] = koopman(observation, data, x0)
    %% Create structure variable for errors
    err = struct;

    %% evaluate for the observation function
    Nx = length(x0(:,1));                   % number of initial points
    M  = round(length(data(:,1))/Nx);       % number of data points
    Nk = length(observation(data(1,:)));    % number of obs. functions

    psiX = NaN(M-Nx, Nk);
    psiY = NaN(M-Nx, Nk);

    i = 0;
    j = 0;

    for n = 1:Nx

        for m = 1:M-1

            i = i + 1;
            j = j + 1;

            psiX(j,:) = observation(data(i,:));
            psiY(j,:) = observation(data(i+1,:));

        end

        i = i + 1;
        j = n*(M-1);

    end

    if (sum(isnan(psiX), 'all') > 0 || sum(isnan(psiY), 'all') > 0)
        err.psiX = psiX;
        err.psiY = psiY;

        K   = NaN;
        acc = NaN;
        ind = NaN;

        fprintf("ERROR: psiX or psiY contain NaN value.\n\n")

        return;
    end
    
    %% perform lest-squares
    % create least-squares matrices
    eps = 1e-6;
    G = 1/M * (psiX')*psiX;
    A = 1/M * (psiX')*psiY;

    [U,S,V] = svd(G);

    ind = diag(S) > eps;

    U = U(:,ind);
    S = S(ind,ind);
    V = V(:,ind);
    
    % solve for the Koopman operator
    K = (V*(S\U')) * A;
    
    % calculate residual error
    acc = 0;
    for n = 1:Nk
    
        acc = acc + norm(psiY(n,:)' - K'*psiX(n,:)');
    
    end
    
    if isnan(acc)

        fprintf("Nk = %i\n", Nk)
        fprintf("G = (%i x %i)\n", size(G))
        disp(G)
        fprintf("A = (%i x %i)\n", size(A))
        disp(A)
        fprintf("K = (%i x %i)\n", size(K))
        disp(K)

        err.G = G;
        err.A = A;
        err.K = K;

    end
end