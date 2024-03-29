clc;
clear;
close all;
b = readmatrix('vectorInObs.txt');
r = readmatrix('vectorInRef.txt');

n = size(b);
n = n(1);

%q = QUEST_mat(b, r, n);

a_i = 1/n;

Z = zeros(1,3); % 1*3 matrix


B = b'*r
B = a_i*B

sigma = .5*trace(B)
S = B + B'
S2 = S*S

for i = 1:n
    Z = Z + cross(b(i,:), r(i,:));
    Z'
end % for i
Z = a_i * Z

K = [S - trace(B)*eye(3) Z'];
K = [K; Z trace(B)];

[eigV, eigD] = eig(K)

D = diag(eigD);
maxEV = max(D);
maxEVIndex = find(D == maxEV, 1);


delta = det(S)
kappa = trace(adjoint(S))

a = sigma*sigma - kappa
bCH = sigma*sigma + Z*Z'
c = delta + (Z*S*Z')
d = Z*S*S*Z'

l = 1;
tol = 1e-12;
max_it = 100;
iters = 0;
error = tol + 1;


while ((error > tol) && (iters < max_it))
    l2 = l - (((l*l*l*l) - (a+bCH)*l*l - c*l + (a*bCH + c*sigma - d)) / (4*l*l*l - 2*l*(a+bCH) -c));
    error = abs(l2 - l);
    l = l2;
    iters = iters + 1;
    
end % end while error > tol && it < max_it
l
l2

beta = l - sigma
alpha = l*l - sigma*sigma + kappa
gamma = (l + sigma)*alpha - delta

vecX = (alpha*eye(3) + beta*S + S*S)*Z'

q_d = sqrt(gamma^2 + sqrt( vecX'*vecX))

qOpt = [vecX(1)/q_d; vecX(2)/q_d; vecX(3)/q_d; gamma/q_d]
q = eigV(:, maxEVIndex)
