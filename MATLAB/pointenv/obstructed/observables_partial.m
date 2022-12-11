function [dPsix, dPsiu] = observables_partial(x, u, world)
%     obs_x = @(x) obsFun(x, u);
%     dPsix = naps.fdm(obs_x, x);
    x1 = x(1);  x2 = x(2);
    u1 = u(1);  u2 = u(2);

    r11 = world(1).x(1);  r21 = world(1).x(2);
    r12 = world(2).x(1);  r22 = world(2).x(2);
    r13 = world(3).x(1);  r23 = world(3).x(2);
    r14 = world(4).x(1);  r24 = world(4).x(2);

    dPsix = [
        1, 0, 2*x1, x2, 0, 0, 0, 0, 0, 0, u1, 0, u2, 0, 2*x1-2*r11, 2*x1-2*r12, 2*x1-2*r13, 2*x1-2*r14;
        0, 1, 0, x1, 2*x2, 0, 0, 0, 0, 0, 0, u1, 0, u2, 2*x2-2*r21, 2*x2-2*r22, 2*x2-2*r23, 2*x2-2*r24;
    ];

%     obs_u = @(u) obsFun(x, u);
%     dPsiu = naps.fdm(obs_u, u);

    dPsiu = [
        0, 0, 0, 0, 0, 1, 0, 2*u1, u2, 0, x1, x2, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 1, 0, u1, 2*u2, 0, 0, x1, x2, 0, 0, 0, 0;
    ];
end