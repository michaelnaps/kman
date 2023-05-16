function [M] = Kblock(K, p, q, b)
    M = [
        eye(p), zeros(p,q*b);
        zeros(q*b,p), kron(eye(q), K)
    ];
end