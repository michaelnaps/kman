%% [K, acc, ind, err] = KoopmanWithControl(observation, xData, x0, uData, eps, depend)
function [K, acc, ind, err] = KoopmanWithControl(observation, xData, x0, uData, eps, depend)
    %% default variables
    [~, meta] = observation(x0(1,:), uData(1,:));      % observables meta-data
    Nk = meta.Nk;

    if nargin < 6
        depend = ones(Nk,1);
    end

    if nargin < 5
        eps = [];
    end


    %% Create structure variable for errors
    TOL = 1e-12;
    err = struct;


    %% evaluate for the observation function
    N0 = length(x0(:,1));                              % number of initial points
    Mx = round(length(xData(:,1))/N0);                 % number of data points

    PsiX = NaN(N0*(Mx-1), Nk);
    PsiY = NaN(N0*(Mx-1), Nk);

    i = 0;
    j = 0;

    for n = 1:N0

        for m = 1:Mx-1

            i = i + 1;
            j = j + 1;

            PsiX(j,:) = observation(xData(i,:), uData(i,:));
            PsiY(j,:) = observation(xData(i+1,:), uData(i+1,:));

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

    if isempty(eps)
        eps = TOL*max(diag(S));
    end

    ind = diag(S) > eps;
    ind = abs(ind+depend-2) < TOL;

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