function [K] = KoopmanAnalytical(world, meta, alpha)

    if nargin < 3
        alpha = 1;
    end

    % coefficient gain
    w = 1;
    
    % term dimensions
    Nx  = length(meta.x);
    Nxx = length(meta.xx);
    Nu  = length(meta.u);
    Nuu = length(meta.uu);
    Nxu = length(meta.xu);
%     Nux = length(meta.ux);
    Nw  = length(meta.d);
    Nk  = meta.Nk;

    % world obstacle center points
    r11 = world(1).x(1);  r12 = world(1).x(2);
    r21 = world(2).x(1);  r22 = world(2).x(2);
    r31 = world(3).x(1);  r32 = world(3).x(2);
    r41 = world(4).x(1);  r42 = world(4).x(2);

    % initialize operator
    K = NaN(Nk);
    
    % state terms: x
    K(meta.x, meta.x) = eye(Nx);
    K(meta.u, meta.x) = eye(Nx);

    % state term expansion: x'x
    K(meta.xx, meta.xx) = w*eye(Nxx);
    K(meta.uu, meta.xx) = w*eye(Nuu);
    K(meta.xu, meta.xx) = w*[
        2, 0, 0;
        0, 1, 0;
        0, 1, 0;
        0, 0, 2;
    ];

   % input terms: u
   K(meta.u, meta.u) = eye(Nu);

   % input term expansion: u'u
   K(meta.uu, meta.uu) = w*eye(Nuu);

   % state-input term expansion: x'u
   K(meta.xu, meta.xu) = w*eye(Nxu,Nxu);
   K(meta.uu, meta.xu) = w*[
       1, 0, 0, 0;
       0, 1, 1, 0;
       0, 0, 0, 1;
   ];

   % distance terms: d(x) = (x - r)(x - r)'
   K(meta.x, meta.d) = -w*2*[
       r11, r21, r31, r41;
       r12, r22, r32, r42
   ];
   K(meta.u, meta.d) = -w*2*[
       r11, r21, r31, r41;
       r12, r22, r32, r42
   ];
   K(meta.xx(1), meta.d) = w*ones(1,Nw);
   K(meta.xx(3), meta.d) = w*ones(1,Nw);

   K(meta.uu(1), meta.d) = w*ones(1,Nw);
   K(meta.uu(3), meta.d) = w*ones(1,Nw);

   K(meta.xu(1), meta.d) = w*2*ones(1,Nw);
   K(meta.xu(4), meta.d) = w*2*ones(1,Nw);

   K(meta.c,meta.d) = [
       r11^2+r12^2, r21^2+r22^2, r31^2+r32^2, r41^2+r42^2
   ];

   % resolve NaN elements
   K(meta.c,meta.c) = 1;
   K(isnan(K)) = 0;
end





















