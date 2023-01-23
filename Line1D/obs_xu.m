function [Psi_xu, meta] = obs_xu(x, u)  
    [Psi_x, meta] = obs_x(x);
    [Psi_u, meta_u] = obs_u(x);

    Psi_xu = [Psi_x; Psi_u*u];

    labels = fieldnames(meta_u);
    for i = 1:length(labels)
        meta.(labels{i}) = meta_u.(labels{i}) + meta.Nk;
    end
end