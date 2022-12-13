function [K] = KoopmanAnalytical(world, META, alpha)

    if nargin < 3
        alpha = 1;
    end
    
    % term dimensions
    Nx  = length(META.x);
    Nxx = length(META.xx);
    Nu  = length(META.u);
    Nuu = length(META.uu);
    Nxu = length(META.xu);
    Nw  = length(META.d);
    Nk  = META.Nk;

    % world obstacle center points
    r11 = world(1).x(1);  r21 = world(1).x(2);
    r12 = world(2).x(1);  r22 = world(2).x(2);
    r13 = world(3).x(1);  r23 = world(3).x(2);
    r14 = world(4).x(1);  r24 = world(4).x(2);

    % initialize operator
    K = NaN(Nk);
    
    % state terms: x
    K(META.x, META.x) = eye(Nx);
    K(META.u, META.x) = eye(Nx);

    % state term expansion: x'x
    K(META.xx, META.xx) = eye(Nxx);
    K(META.uu, META.xx) = eye(Nuu);
    K(META.xu, META.xx) = 2*eye(Nxu);

   % input terms: u
   K(META.u, META.u) = eye(Nu);

   % input term expansion: u'u
   K(META.uu, META.uu) = eye(Nuu);

   % input-state term expansion: x'u
   K(META.xu, META.xu) = eye(Nxu);
   K(META.uu, META.xu) = eye(Nuu);

   % distance terms: d(x) = (x - r)(x - r)'
   K(META.x, META.d) = -2*[
       r11, r12, r13, r14;
       r21, r22, r23, r24
   ];
   K(META.xx, META.d) = eye(Nxx);
   K(META.u, META.d) = -2*[
       r11, r12, r13, r14;
       r21, r22, r23, r24
   ];
   K(META.uu, META.d) = eye(Nuu);
   K(META.xu, META.d) = 2*eye(Nxu);
   K(META.c, META.d) = -[
       (r11^2 + r21^2), (r12^2 + r22^2), (r13^2 + r23^2), (r14^2 + r24^2)
   ];

   % constant term and resolve NaN elements
   K(META.c, META.c) = 1;
   K(isnan(K)) = 0;
end