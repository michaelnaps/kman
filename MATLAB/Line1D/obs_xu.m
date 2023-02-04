function [Psi_xu, meta] = obs_xu(X)  
    x = X(1:2);  u = X(3);

    [Psi_x, meta_x] = obs_x(x);
    [Psi_u, meta_u] = obs_u(x);
    [Psi_h, meta_h] = obs_h(X);

    Psi_xu = [Psi_x; kron(Psi_u, Psi_h)];

    meta = meta_x;

    labels_h = fieldnames(meta_h);
    for i = 1:length(labels_h)
        meta_x.(labels_h{i}) = meta_h.(labels_h{i}) + meta_x.Nk;
    end
    
    meta.Nk = meta.Nk + meta_u.Nk*meta_h.Nk;
end