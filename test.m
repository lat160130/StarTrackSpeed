clc;
clear;

% we are reading in the matrices as column vectors
b = readmatrix('vectorInObs.txt')';
r = readmatrix('vectorInRef.txt')';
% x
% y
% z

n = size(b);
n = n(2);
a_i = 1/n;


B = zeros(3); % 3x3 matrix
z = zeros(1,3); % 1*3 matrix
for i = 1:n
    mat = a_i* (b(:,i)' * r(:,i));
    B = B + mat;

    matCr = a_i * cross(b(:,i), r(:,i));
    z = z + matCr;

end % 

