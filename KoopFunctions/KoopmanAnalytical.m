function [K] = KoopmanAnalytical(world, META, alpha)

    if nargin < 3
        alpha = 1;
    end

    % coefficient gain
    w = 1;
    
    % term dimensions
    Nx  = length(META.x);
    Nxx = length(META.xx);
    Nu  = length(META.u);
    Nuu = length(META.uu);
    Nxu = length(META.xu);
%     Nux = length(META.ux);
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
    K(META.xx, META.xx) = w*eye(Nxx);
    K(META.uu, META.xx) = w*eye(Nuu);
    K(META.xu(1), META.xx) = w*[2,0,0];
    K(META.xu(2), META.xx) = w*[0,1,0];
    K(META.xu(3), META.xx) = w*[0,1,0];
    K(META.xu(4), META.xx) = w*[0,0,2];

   % input terms: u
   K(META.u, META.u) = eye(Nu);

   % input term expansion: u'u
   K(META.uu, META.uu) = w*eye(Nuu);

   % state-input term expansion: x'u
   K(META.xu, META.xu) = w*eye(Nxu);
   K(META.uu, META.xu) = w*eye(Nuu,Nxu);

   % distance terms: d(x) = (x - r)(x - r)'
   K(META.d, META.d) = w*eye(Nw);
   K(META.u, META.d) = -w*2*[
       r11, r12, r13, r14;
       r21, r22, r23, r24
   ];
   K(META.uu(1), META.d) = w*ones(1,Nw);
   K(META.uu(3), META.d) = w*ones(1,Nw);
   K(META.xu(1), META.d) = w*ones(1,Nw);
   K(META.xu(3), META.d) = w*ones(1,Nw);

   % resolve NaN elements
   K(isnan(K)) = 0;
end





















