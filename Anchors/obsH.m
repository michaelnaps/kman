function [PsiH, metaH] = obsH(X)
    load anchors anchors;

    x = X(1:4);
    u = X(5:6);

    dist = NaN(4,1);
    for i = 1:4
        dist(i) = (x(1:2) - anchors(i).x)'*(x(1:2) - anchors(i).x);
    end

    PsiH = [u; x; dist; 1];

    metaH.u = 1:2;
    metaH.x = 3:6;
    metaH.a = 7:10;
    metaH.c = 11;
    metaH.Nk = 11;
end