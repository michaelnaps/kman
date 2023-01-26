function [PsiH, metaH] = obsH(X)
    load anchors anchors;

    x = X(1:4);
    u = X(5:6);

    dist = NaN(4,1);
    for i = 1:4
        dist(i) = (x(1:2) - anchors(i).x)'*(x(1:2) - anchors(i).x);
    end

    PsiH = [u; dist];

    metaH.u = 1:2;
    metaH.a = 1:4 + metaH.u(end);
    metaH.c = 1 + metaH.a(end);
    metaH.Nk = 7 - 1;
end