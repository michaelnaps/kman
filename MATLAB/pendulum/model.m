function qn = model(q, dt)
    g = 9.81;
    l = 1;
    c = 1;

    % pendulum dynamics function (discrete)
    qn = q + dt * [
        q(2), -(g/l)*cos(q(1)) - c*q(2)
    ];
end