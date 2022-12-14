function [K, acc, ind, err] = KoopmanWithControl(observation, x_data, x0, u_data, eps)
    %% create structure variable for errors
    err = struct;

    %% evaluate for the observation function
    N0 = length(x0(:,1));                                % number of initial points
    Mx = round(length(x_data(:,1))/N0);                  % number of data points
    [~, META] = observation(x0(1,:), u_data(1,:));      % observables meta-data
    Nk = META.Nk;

    Nu = length(META.u);
    PsiX = NaN(N0*(Mx-1), Nk);
    PsiY = NaN(N0*(Mx-1), Nk);

    i = 0;
    j = 0;

    for n = 1:N0

        for m = 1:Mx-1

            i = i + 1;
            j = j + 1;

            PsiX(j,:) = observation(x_data(i,:), u_data(i,:));
            PsiY(j,:) = observation(x_data(i+1,:), u_data(i+1,:));
%             PsiY(j,:) = observation(x_data(i+1,:), zeros(1,Nu));

        end

        i = i + 1;
        j = n*(Mx-1);

    end

    if (sum(isnan(PsiX), 'all') > 0 || sum(isnan(PsiY), 'all') > 0)
        err.PsiX = PsiX;
        err.PsiY = PsiY;

        K   = NaN;
        acc = NaN;
        ind = NaN;

        fprintf("ERROR: PsiX or PsiY contain NaN value.\n\n")

        return;
    end
    
    %% perform lest-squares
    % create least-squares matrices
    % (according to abraham, model-based)
    G = 1/(N0*(Mx-1)) * (PsiX')*PsiX;
    A = 1/(N0*(Mx-1)) * (PsiX')*PsiY;

    [U,S,V] = svd(G);

    if nargin < 5
        eps = 1e-12*max(diag(S));
    end

    ind = diag(S) > eps;

    U = U(:,ind);
    S = S(ind,ind);
    V = V(:,ind);
    
    % solve for the Koopman operator
    K = (V*(S\U')) * A;
    
    % calculate residual error
    acc = 0;
    for n = 1:N0*(Mx-1)    
        acc = acc + norm(PsiY(n,:) - PsiX(n,:)*K);
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

    else

        err.U = U;
        err.S = S;
        err.V = V;
        err.eps = eps;

    end

end