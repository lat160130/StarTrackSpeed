clc;
clear;

% we are reading in the matrices as column vectors
a3 = readmatrix('vectorInObs.txt');
b3 = readmatrix('vectorInRef.txt');

n = size(a3);
n = n(1);

a_i = 1 / n;
% x
% y
% z

% n = size(b);
% n = n(2);
% a_i = 1/n;


% B = zeros(3); % 3x3 matrix
% z = zeros(1,3); % 1*3 matrix
% for i = 1:n
%     mat = a_i* (b(:,i)' * r(:,i));
%     B = B + mat;
% 
%     matCr = a_i * cross(b(:,i), r(:,i));
%     z = z + matCr;
% 
% end % 


B = a3'*b3;
S = B + B';

Z = zeros(1,3);

for i = 1:n
    a3(i,:)
    b3(i,:)
    Z = Z + a_i*cross(a3(i,:), b3(i,:));
end

sigma = .5*trace(S);
delta = det(S);
kappa = trace(adjoint(S));
a = sigma^2 - kappa;
b = Z*Z' + sigma^2;
c = delta + Z*S*Z'
d = Z*S*S*Z'

