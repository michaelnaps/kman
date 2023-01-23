%% [K, acc, ind, err] = Koopman(observables, X, Y, x0, eps)
function [K, acc, ind, err] = Koopman(observables, X, Y, x0, eps)
    %% default variables
    [~, meta] = observables(x0(:,1));
    Nk = meta.Nk;

    if nargin < 5
        eps = [];
    end


    %% Create structure variable for errors
    TOL = 1e-12;
    err = struct;


    %% evaluate for the observation function
    N0 = length(x0(1,:));                          % number of initial points
    Mx = round(length(X(1,:))/N0);                 % number of data points

    PsiX = NaN(Nk, N0*(Mx-1));
    PsiY = NaN(Nk, N0*(Mx-1));

    i = 0;
    j = 0;

    for n = 1:N0

        for m = 1:Mx-1

            i = i + 1;
            j = j + 1;

            PsiX(:,j) = observables(X(:,i));
            PsiY(:,j) = observables(Y(:,i));

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
    G = 1/(N0*(Mx-1)) * PsiX*(PsiX)';
    A = 1/(N0*(Mx-1)) * PsiX*(PsiY)';

    [U,S,V] = svd(G);

    if isempty(eps)
        eps = TOL*max(diag(S));
    end

    ind = diag(S) > eps;

    U = U(:,ind);
    S = S(ind,ind);
    V = V(:,ind);
    
    % solve for the Koopman operator
    K = (V*(S\U')) * A;
    K = K';
    
    % calculate residual error
    acc = 0;
    for n = 1:N0*(Mx-1)    
        acc = acc + norm(PsiY(:,n) - K*PsiX(:,n));
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